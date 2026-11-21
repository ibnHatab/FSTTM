// Package hometools is the protocol between the conversation's LLM and the home's rule
// engine: the rule engine (the Metis home service; fsttm/feeds/metis.py in Python) serves a
// local TCP port, 127.0.0.1:8768 by default, one JSON request per line, one JSON answer per
// line:
//
//	{"op": "tools"}                                   → {"tools": [<OpenAI function tool>, ...]}
//	{"op": "call", "domain": "home_air", "args": {"intent": "ON", "device": "lab ventilation"}}
//	                                                  → {"status": "done"|"refused"|"pending"|"error", "text": "..."}
//
// One tool per package of the home (home_lights, home_air, ...): its intent schema as the
// function's parameters. A call hands the intent to the rules and answers with their
// outcome; the LLM says it. Client is the engine's side, Serve the rule engine's.
package hometools

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"strings"
	"sync"
	"time"
)

// DefaultAddr is the rule engine's tool port.
const DefaultAddr = "127.0.0.1:8768"

// Tool is an OpenAI-style function tool.
type Tool struct {
	Type     string   `json:"type"` // "function"
	Function Function `json:"function"`
}

type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// Request is one line in.
type Request struct {
	Op     string         `json:"op"` // "tools" | "call"
	Domain string         `json:"domain,omitempty"`
	Args   map[string]any `json:"args,omitempty"`
}

// Outcome is a call's answer.
type Outcome struct {
	Status string `json:"status"` // done | refused | pending | error
	Text   string `json:"text"`
}

// ── the engine's side ─────────────────────────────────────────────────────────────────

// Client asks the rule engine. The tool list is cached, and asked again (at most every
// Retry, 30 s by default) while the rule engine is not there.
type Client struct {
	Addr    string
	Retry   time.Duration
	Timeout time.Duration // per call, 10 s by default
	mu      sync.Mutex
	tools   []json.RawMessage
	at      time.Time
}

func (c *Client) roundTrip(req Request, timeout time.Duration) ([]byte, error) {
	conn, err := net.DialTimeout("tcp", c.Addr, 3*time.Second)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(timeout))
	b, _ := json.Marshal(req)
	if _, err := conn.Write(append(b, '\n')); err != nil {
		return nil, err
	}
	return bufio.NewReader(conn).ReadBytes('\n')
}

// Tools is the tool list as raw JSON (to pass to an LLM request as is), nil when the rule
// engine is not there.
func (c *Client) Tools() []json.RawMessage {
	if c == nil || c.Addr == "" {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	retry := c.Retry
	if retry <= 0 {
		retry = 30 * time.Second
	}
	if c.tools == nil && (c.at.IsZero() || time.Since(c.at) > retry) {
		c.at = time.Now()
		raw, err := c.roundTrip(Request{Op: "tools"}, 5*time.Second)
		if err != nil {
			log.Printf("[hometools] no rule engine at %s yet (%v)", c.Addr, err)
			return nil
		}
		var out struct {
			Tools []json.RawMessage `json:"tools"`
		}
		if json.Unmarshal(raw, &out) == nil && len(out.Tools) > 0 {
			c.tools = out.Tools
		}
	}
	return c.tools
}

// Call hands an intent to a package (name with or without "home_") and returns the
// outcome; the JSON answer line is returned too, to feed back to the LLM verbatim.
func (c *Client) Call(name string, args map[string]any) (Outcome, string) {
	timeout := c.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	raw, err := c.roundTrip(Request{Op: "call", Domain: name, Args: args}, timeout)
	if err != nil {
		c.mu.Lock()
		c.tools = nil // the rule engine went away: ask again later
		c.mu.Unlock()
		o := Outcome{Status: "error", Text: "the rule engine is not reachable: " + err.Error()}
		b, _ := json.Marshal(o)
		return o, string(b)
	}
	line := strings.TrimSpace(string(raw))
	var o Outcome
	if err := json.Unmarshal(raw, &o); err != nil {
		o = Outcome{Status: "error", Text: "bad answer: " + line}
	}
	return o, line
}

// ── the rule engine's side ────────────────────────────────────────────────────────────

// Handler is what a rule engine provides: its tools, and a call's outcome.
type Handler interface {
	Tools() []Tool
	Call(domain string, args map[string]any) Outcome
}

// Serve answers the protocol on addr until the listener is closed (returned, so the
// caller can close it). Each connection may send several requests.
func Serve(addr string, h Handler) (net.Listener, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				sc := bufio.NewScanner(conn)
				sc.Buffer(make([]byte, 1<<16), 1<<20)
				for sc.Scan() {
					var req Request
					var out any
					if err := json.Unmarshal(sc.Bytes(), &req); err != nil {
						out = Outcome{Status: "error", Text: err.Error()}
					} else if req.Op == "tools" {
						out = map[string]any{"tools": h.Tools()}
					} else if req.Op == "call" {
						out = h.Call(strings.TrimPrefix(req.Domain, "home_"), req.Args)
					} else {
						out = Outcome{Status: "error", Text: fmt.Sprintf("op is tools or call, not %q", req.Op)}
					}
					b, _ := json.Marshal(out)
					if _, err := conn.Write(append(b, '\n')); err != nil {
						return
					}
				}
			}()
		}
	}()
	return ln, nil
}
