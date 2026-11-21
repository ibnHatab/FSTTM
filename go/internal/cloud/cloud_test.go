package cloud

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ibnHatab/fsttm/go/pkg/hometools"
)

// fake is an OpenAI-compatible endpoint: it streams `reply`, or first a tool call when
// `tool` is set and the request has tools and no tool message yet; `busy` 503s come first.
type fake struct {
	mu     sync.Mutex
	bodies []map[string]any
	reply  []string
	tool   [2]string // name, arguments
	busy   int
	think  bool
}

func (f *fake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	_ = json.NewDecoder(r.Body).Decode(&body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	busy := f.busy > 0
	if busy {
		f.busy--
	}
	f.mu.Unlock()
	if busy {
		w.WriteHeader(503)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	send := func(v any) { b, _ := json.Marshal(v); fmt.Fprintf(w, "data: %s\n\n", b) }
	hasTool := false
	for _, m := range body["messages"].([]any) {
		if m.(map[string]any)["role"] == "tool" {
			hasTool = true
		}
	}
	if f.tool[0] != "" && body["tools"] != nil && !hasTool {
		send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "id": "c1", "extra_content": map[string]any{"sig": "x"},
				"function": map[string]any{"name": f.tool[0], "arguments": f.tool[1]}}}}}}})
	} else if f.think {
		send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "hmm"},
			"finish_reason": "length"}}})
	} else {
		for _, p := range f.reply {
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": p}}}})
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func toolPort(t *testing.T, answer string) (string, *[]map[string]any) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	seen := &[]map[string]any{}
	var mu sync.Mutex
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				r := bufio.NewReader(c)
				line, _ := r.ReadBytes('\n')
				var req map[string]any
				_ = json.Unmarshal(line, &req)
				mu.Lock()
				*seen = append(*seen, req)
				mu.Unlock()
				if req["op"] == "tools" {
					fmt.Fprintln(c, `{"tools": [{"type": "function", "function": {"name": "home_air", "description": "air", "parameters": {"type": "object"}}}]}`)
				} else {
					fmt.Fprintln(c, answer)
				}
			}()
		}
	}()
	return ln.Addr().String(), seen
}

func agent(url string, tools *hometools.Client) *Agent {
	return &Agent{Client: NewClient(url, "m", "k", map[string]any{"reasoning_effort": "none"}, 5*time.Second),
		Streams: NewStreams(), Tools: tools, Persona: "You are Nina.", MaxTokens: 200}
}

func TestChatStreamsSentencesAndKeepsHistory(t *testing.T) {
	f := &fake{reply: []string{"Hello", " there. How", " are you?"}}
	s := httptest.NewServer(f)
	defer s.Close()
	a := agent(s.URL, nil)
	var said []string
	full, err := a.Turn(context.Background(), "hi", func(x string) { said = append(said, x) })
	if err != nil || full != "Hello there. How are you?" {
		t.Fatalf("full %q err %v", full, err)
	}
	if strings.Join(said, "|") != "Hello there.|How are you?" {
		t.Fatalf("sentences %q", said)
	}
	_, _ = a.Turn(context.Background(), "again", func(string) {})
	msgs := f.bodies[1]["messages"].([]any)
	if len(msgs) != 4 || msgs[2].(map[string]any)["content"] != "Hello there. How are you?" {
		t.Fatalf("history not kept: %v", msgs)
	}
	if f.bodies[0]["reasoning_effort"] != "none" || f.bodies[0]["stream"] != true {
		t.Fatalf("body %v", f.bodies[0])
	}
	sys := msgs[0].(map[string]any)["content"].(string)
	if !strings.HasPrefix(sys, "You are Nina.") || !strings.Contains(sys, "three separate streams") {
		t.Fatalf("system prompt %q", sys)
	}
}

func TestStreamsInSystemPromptNotTurns(t *testing.T) {
	f := &fake{reply: []string{"Yes."}}
	s := httptest.NewServer(f)
	defer s.Close()
	a := agent(s.URL, nil)
	off, on := "off", "on"
	a.Streams.Env("light.x", &off, time.Unix(1, 0))
	a.Streams.Env("light.x", &on, time.Unix(2, 0))
	a.Streams.Reason("refused on(heating): not a device", time.Unix(3, 0))
	_, _ = a.Turn(context.Background(), "is the light on?", func(string) {})
	msgs := f.bodies[0]["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("the streams became turns: %v", msgs)
	}
	sys := msgs[0].(map[string]any)["content"].(string)
	if !strings.Contains(sys, "- light.x: on") || strings.Contains(sys, "light.x: off") ||
		!strings.Contains(strings.SplitN(sys, "REASONING (newest last):", 2)[1], "refused on(heating)") {
		t.Fatalf("system prompt %q", sys)
	}
}

func TestToolRoundToTheRuleEngine(t *testing.T) {
	f := &fake{reply: []string{"The lab ventilation is stopped."}, tool: [2]string{"home_air", `{"intent":"NIGHT","device":"lab ventilation"}`}}
	s := httptest.NewServer(f)
	defer s.Close()
	addr, seen := toolPort(t, `{"status": "refused", "text": "the lab ventilation is stopped."}`)
	a := agent(s.URL, &hometools.Client{Addr: addr})
	full, err := a.Turn(context.Background(), "night mode in the lab", func(string) {})
	if err != nil || full != "The lab ventilation is stopped." {
		t.Fatalf("full %q err %v", full, err)
	}
	if len(*seen) != 2 || (*seen)[0]["op"] != "tools" || (*seen)[1]["domain"] != "home_air" {
		t.Fatalf("sidecar saw %v", *seen)
	}
	second := f.bodies[1]["messages"].([]any)
	last := second[len(second)-1].(map[string]any)
	if last["role"] != "tool" || !strings.Contains(last["content"].(string), "refused") {
		t.Fatalf("tool message %v", last)
	}
	call := second[len(second)-2].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if call["extra_content"] == nil || call["function"].(map[string]any)["name"] != "home_air" {
		t.Fatalf("the call was not sent back as it came: %v", call)
	}
	if !strings.Contains(f.bodies[0]["messages"].([]any)[0].(map[string]any)["content"].(string), "home_lights") {
		t.Fatal("the tools paragraph is not in the prompt")
	}
}

func TestBusyProviderIsRetried(t *testing.T) {
	f := &fake{reply: []string{"Hi."}, busy: 2}
	s := httptest.NewServer(f)
	defer s.Close()
	full, err := agent(s.URL, nil).Turn(context.Background(), "hi", func(string) {})
	if err != nil || full != "Hi." {
		t.Fatalf("full %q err %v", full, err)
	}
}

func TestThinkingOnlyReplySaysHowToTurnItOff(t *testing.T) {
	f := &fake{think: true}
	s := httptest.NewServer(f)
	defer s.Close()
	_, err := agent(s.URL, nil).Turn(context.Background(), "hi", func(string) {})
	if err == nil || !strings.Contains(err.Error(), "thinking") {
		t.Fatalf("err %v", err)
	}
}

func TestSentenceBuf(t *testing.T) {
	var s SentenceBuf
	got := append(s.Add("One. Tw"), s.Add("o! Three")...)
	if strings.Join(got, "|") != "One.|Two!" || s.Flush() != "Three" {
		t.Fatalf("%q", got)
	}
}
