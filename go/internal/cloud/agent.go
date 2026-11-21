package cloud

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ibnHatab/fsttm/go/pkg/hometools"
)

// Streams are the two streams beside the conversation: env (keyed, a newer entry
// replacing the older; nil text removes the key) and reasoning (appended, the last 12).
type Streams struct {
	mu        sync.Mutex
	env       map[string]entry
	reasoning []entry
	Keep      int
}

type entry struct {
	Text string
	TS   time.Time
}

func NewStreams() *Streams { return &Streams{env: map[string]entry{}, Keep: 12} }

// Env sets (text non-nil) or removes (nil) an entity's line.
func (s *Streams) Env(key string, text *string, ts time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if text == nil {
		delete(s.env, key)
		return
	}
	s.env[key] = entry{*text, ts}
}

// Reason appends a conclusion of the rule engine.
func (s *Streams) Reason(text string, ts time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reasoning = append(s.reasoning, entry{text, ts})
	if len(s.reasoning) > s.Keep {
		s.reasoning = s.reasoning[len(s.reasoning)-s.Keep:]
	}
}

// Render is the ENVIRONMENT and REASONING sections of the system prompt ("" when empty),
// as fsttm/cloud.py's render_streams writes them.
func (s *Streams) Render(now time.Time) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	hm := func(t time.Time) string {
		if t.IsZero() {
			return "--:--:--"
		}
		return t.Format("15:04:05")
	}
	var out []string
	if len(s.env) > 0 {
		out = append(out, "ENVIRONMENT (as of "+hm(now)+"):")
		keys := make([]string, 0, len(s.env))
		for k := range s.env {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			e := s.env[k]
			out = append(out, fmt.Sprintf("- %s: %s  (%s)", k, e.Text, hm(e.TS)))
		}
	}
	if len(s.reasoning) > 0 {
		out = append(out, "REASONING (newest last):")
		for _, e := range s.reasoning {
			out = append(out, fmt.Sprintf("- %s %s", hm(e.TS), e.Text))
		}
	}
	return strings.Join(out, "\n")
}

func (a *Agent) call(c ToolCall) string {
	var args map[string]any
	if err := json.Unmarshal([]byte(orEmpty(c.Function.Arguments)), &args); err != nil {
		return `{"status": "error", "text": "the arguments are not JSON"}`
	}
	_, line := a.Tools.Call(c.Function.Name, args)
	return line
}

func orEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "{}"
	}
	return s
}

// Agent is the conversation: history, the streams, the tools, one client.
type Agent struct {
	Client    *Client
	Streams   *Streams
	Tools     *hometools.Client // the rule engine's tools (nil: none)
	Persona   string // the base system prompt (VoiceStyle is added after it)
	Temp      float64
	MaxTokens int
	KeepTurns int

	mu    sync.Mutex
	turns [][2]string
	// OnTool, when set, sees every tool call and its answer (the panel, the log).
	OnTool func(name, args, answer string)
}

func (a *Agent) system() string {
	base := a.Persona
	if base != "" {
		base += "\n\n" + VoiceStyle
	} else {
		base = VoiceStyle
	}
	parts := []string{base, StreamsPrompt}
	if len(a.Tools.Tools()) > 0 {
		parts = append(parts, ToolsPrompt)
	}
	if st := a.Streams.Render(time.Now()); st != "" {
		parts = append(parts, st)
	}
	return strings.Join(parts, "\n\n")
}

// Turn answers one person turn. onSentence gets each complete sentence as the reply
// streams in (the speaker's unit); the full reply comes back at the end and joins the
// history. A cancelled ctx (barge-in) ends it; what was written stays in the history (no
// rollback with an API: there is no KV cache holding the unsaid part).
func (a *Agent) Turn(ctx context.Context, text string, onSentence func(string)) (string, error) {
	a.mu.Lock()
	msgs := []Message{Text("system", a.system())}
	for _, t := range a.turns {
		msgs = append(msgs, Text("user", t[0]), Text("assistant", t[1]))
	}
	a.mu.Unlock()
	msgs = append(msgs, Text("user", text))

	var acc strings.Builder
	var sb SentenceBuf
	emit := func(p string) {
		acc.WriteString(p)
		for _, s := range sb.Add(p) {
			onSentence(s)
		}
	}
	kw := map[string]any{"max_tokens": a.MaxTokens}
	if a.Temp > 0 {
		kw["temperature"] = a.Temp
	}
	var err error
	for round := 0; round < 3; round++ {
		tl := a.Tools.Tools()
		var calls []ToolCall
		calls, err = a.Client.Stream(ctx, msgs, tl, kw, emit)
		if err != nil || len(calls) == 0 || ctx.Err() != nil {
			break
		}
		for i := range calls {
			if calls[i].ID == "" {
				calls[i].ID = fmt.Sprintf("call_%d_%d", round, i)
			}
		}
		said := acc.String()
		am := Message{Role: "assistant", ToolCalls: calls}
		if said != "" {
			am.Content = &said
		}
		msgs = append(msgs, am)
		for _, c := range calls {
			ans := a.call(c)
			log.Printf("[cloud] tool %s(%s) -> %s", c.Function.Name, c.Function.Arguments, ans)
			if a.OnTool != nil {
				a.OnTool(c.Function.Name, c.Function.Arguments, ans)
			}
			tm := Text("tool", ans)
			tm.ToolCallID = c.ID
			msgs = append(msgs, tm)
		}
		if said != "" && !strings.HasSuffix(said, " ") {
			emit(" ")
		}
	}
	if rest := sb.Flush(); rest != "" {
		onSentence(rest)
	}
	full := strings.TrimSpace(acc.String())
	if full != "" {
		a.mu.Lock()
		a.turns = append(a.turns, [2]string{text, full})
		keep := a.KeepTurns
		if keep <= 0 {
			keep = 16
		}
		if len(a.turns) > keep {
			a.turns = a.turns[len(a.turns)-keep:]
		}
		a.mu.Unlock()
	}
	return full, err
}

// SentenceBuf cuts streamed text into sentences: a sentence ends at . ! ? followed by
// whitespace (or a newline).
type SentenceBuf struct{ buf strings.Builder }

func (s *SentenceBuf) Add(p string) []string {
	s.buf.WriteString(p)
	t := s.buf.String()
	var out []string
	start := 0
	for i := 0; i < len(t)-1; i++ {
		c := t[i]
		if (c == '.' || c == '!' || c == '?' || c == '\n') && (t[i+1] == ' ' || t[i+1] == '\n') {
			if seg := strings.TrimSpace(t[start : i+1]); seg != "" {
				out = append(out, seg)
			}
			start = i + 1
		}
	}
	s.buf.Reset()
	s.buf.WriteString(t[start:])
	return out
}

func (s *SentenceBuf) Flush() string {
	t := strings.TrimSpace(s.buf.String())
	s.buf.Reset()
	return t
}
