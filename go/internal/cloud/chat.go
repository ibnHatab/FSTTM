// Package cloud is the LLM over an OpenAI-compatible chat API (Gemini, OpenRouter, a
// llama-server): the Go port of fsttm/cloud.py. A streamed reply is spoken sentence by
// sentence; the conversation is a message list (no KV cache to prime or roll back); the
// home's ENVIRONMENT and REASONING streams are rendered into the system prompt; the home's
// tools come from the Metis sidecar's port (fsttm-metis) and their calls are run and fed
// back for the spoken reply.
package cloud

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// A busy provider answers 429 / 5xx ("high demand"); a short wait usually gets through.
var (
	RetryCodes = map[int]bool{429: true, 500: true, 502: true, 503: true}
	RetryWaits = []time.Duration{600 * time.Millisecond, 1500 * time.Millisecond}
)

const thinkingHint = "the model spent its whole reply on thinking (finish_reason length, no text): " +
	"turn thinking off with request_extra, e.g. {reasoning_effort: none} (OpenAI-style APIs; " +
	"Gemini: the lowest effort the model accepts) or, for a llama-server, " +
	"{chat_template_kwargs: {enable_thinking: false}}; or raise max_tokens"

// Message is one chat message (system, user, assistant with tool calls, tool result).
type Message struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Text is a message with content.
func Text(role, content string) Message { return Message{Role: role, Content: &content} }

// ToolCall is a call as the provider streamed it; Extra keeps whatever else it attached
// (Gemini's thought signature), sent back as it came.
type ToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function ToolFunction   `json:"function"`
	Extra    map[string]any `json:"-"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (c ToolCall) MarshalJSON() ([]byte, error) {
	m := map[string]any{}
	for k, v := range c.Extra {
		m[k] = v
	}
	m["id"], m["type"], m["function"] = c.ID, c.Type, c.Function
	return json.Marshal(m)
}

// Client is one endpoint: chat/completions, streamed.
type Client struct {
	URL     string
	Model   string
	Key     string
	Extra   map[string]any // merged into every request body
	Timeout time.Duration
	HTTP    *http.Client
}

func NewClient(baseURL, model, key string, extra map[string]any, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &Client{URL: strings.TrimRight(baseURL, "/") + "/chat/completions", Model: model,
		Key: key, Extra: extra, Timeout: timeout, HTTP: &http.Client{Timeout: timeout}}
}

// KeyFrom reads the key: a file (~ expanded) wins, else the environment variable.
func KeyFrom(file, env string) string {
	if file != "" {
		if strings.HasPrefix(file, "~/") {
			file = os.Getenv("HOME") + file[1:]
		}
		if b, err := os.ReadFile(file); err == nil {
			return strings.TrimSpace(string(b))
		}
	}
	if env == "" {
		env = "GEMINI_API_KEY"
	}
	return os.Getenv(env)
}

func (c *Client) body(msgs []Message, stream bool, kw map[string]any) ([]byte, error) {
	b := map[string]any{"model": c.Model, "messages": msgs, "stream": stream}
	for k, v := range c.Extra {
		b[k] = v
	}
	for k, v := range kw {
		if v != nil {
			b[k] = v
		}
	}
	return json.Marshal(b)
}

func (c *Client) open(ctx context.Context, body []byte) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST", c.URL, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if c.Key != "" {
			req.Header.Set("Authorization", "Bearer "+c.Key)
		}
		resp, err := c.HTTP.Do(req)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", c.URL, err)
		}
		if resp.StatusCode == 200 {
			return resp, nil
		}
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		resp.Body.Close()
		if RetryCodes[resp.StatusCode] && attempt < len(RetryWaits) {
			log.Printf("[cloud] HTTP %d: again in %v", resp.StatusCode, RetryWaits[attempt])
			select {
			case <-time.After(RetryWaits[attempt]):
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return nil, fmt.Errorf("HTTP %d from %s: %s", resp.StatusCode, c.URL, detail)
	}
}

// Stream sends the messages and calls onPiece for each piece of text as it arrives. Tool
// calls in the stream are gathered and returned. A reply that is only thinking is an error
// that says how to turn it off.
func (c *Client) Stream(ctx context.Context, msgs []Message, tools []json.RawMessage,
	kw map[string]any, onPiece func(string)) ([]ToolCall, error) {
	if kw == nil {
		kw = map[string]any{}
	}
	if len(tools) > 0 {
		kw["tools"] = tools
	}
	body, err := c.body(msgs, true, kw)
	if err != nil {
		return nil, err
	}
	resp, err := c.open(ctx, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	said, thought, finish := false, false, ""
	byIdx := map[int]*ToolCall{}
	order := []int{}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<16), 1<<22)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[5:])
		if data == "[DONE]" {
			break
		}
		var ev struct {
			Error   any `json:"error"`
			Choices []struct {
				Delta        map[string]json.RawMessage `json:"delta"`
				FinishReason string                     `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if ev.Error != nil {
			return nil, fmt.Errorf("stream error: %v", ev.Error)
		}
		for _, ch := range ev.Choices {
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			d := ch.Delta
			if r, ok := d["reasoning_content"]; ok && string(r) != "null" && string(r) != `""` {
				thought = true
			}
			if r, ok := d["reasoning"]; ok && string(r) != "null" && string(r) != `""` {
				thought = true
			}
			if raw, ok := d["tool_calls"]; ok {
				var tcs []map[string]json.RawMessage
				_ = json.Unmarshal(raw, &tcs)
				for i, tc := range tcs {
					idx := i
					if v, ok := tc["index"]; ok {
						_ = json.Unmarshal(v, &idx)
					}
					cur := byIdx[idx]
					if cur == nil {
						cur = &ToolCall{Type: "function", Extra: map[string]any{}}
						byIdx[idx] = cur
						order = append(order, idx)
					}
					for k, v := range tc {
						switch k {
						case "index":
						case "id":
							_ = json.Unmarshal(v, &cur.ID)
						case "type":
							_ = json.Unmarshal(v, &cur.Type)
						case "function":
							var fn struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							}
							_ = json.Unmarshal(v, &fn)
							if cur.Function.Name == "" {
								cur.Function.Name = fn.Name
							}
							cur.Function.Arguments += fn.Arguments
						default:
							var x any
							_ = json.Unmarshal(v, &x)
							if x != nil {
								cur.Extra[k] = x
							}
						}
					}
				}
			}
			if raw, ok := d["content"]; ok {
				var piece string
				if json.Unmarshal(raw, &piece) == nil && piece != "" {
					said = true
					onPiece(piece)
				}
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}
	if err := sc.Err(); err != nil && ctx.Err() == nil {
		return nil, err
	}
	calls := make([]ToolCall, 0, len(order))
	for _, i := range order {
		calls = append(calls, *byIdx[i])
	}
	if !said && len(calls) == 0 && (thought || finish == "length") {
		return nil, errors.New(thinkingHint)
	}
	return calls, nil
}
