// Package panel serves the engine's page: the dialog (fsttm-web's panel) in the metis
// consoles' shell, with the ENVIRONMENT and REASONING streams in docks beside it, over one
// websocket. The page is embedded (web/); the engine pushes frames, the page sends typed
// person turns.
//
// The wire (JSON text frames, each with "ev"), the application's own frames of fsttm-web's
// wire (its README: "An application extends the wire"):
//
//	engine → page  nuc/snapshot{env, reasoning, conversation, state}   on connect
//	               nuc/env{key, text, ts}        an entity's line (text null: gone)
//	               nuc/reasoning{text, ts}       a conclusion of the rule engine
//	               nuc/tool{name, args, answer}  a tool call the LLM made, and the rules' answer
//	               nuc/heard{text, ts}           what the mic heard and the engine took as a turn
//	               nuc/asleep{text, ts}          heard while asleep (no wake word): ignored
//	               nuc/reply{id?, text, ts}      her whole reply (id: a typed turn's)
//	               nuc/state{floor, attention, dialog}
//	               nuc/event{kind, t, ...}       a dialog event: kind fsttm {event, src, dst} is a floor
//	                                             transition; the rest (stt, llm, speak, cut, echo, heard,
//	                                             asleep, reply, dialog, typed) the engine's own
//	               nuc/doubtful{text, ts}        a transcript that looped (a phrase over and over): not a turn
//	page → engine  nuc/say{id, text}             a typed person turn
//	               nuc/ctl{op}                   a switch: "mic on" | "mic off" | "voice on" | "voice off" | "stop"
//	                                             (hush the narration now); the state comes back as nuc/state{mic, voice}
package panel

import (
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"golang.org/x/net/websocket"
)

//go:embed web
var webFS embed.FS

type Server struct {
	OnSay func(id, text string) // a typed person turn
	OnCtl func(op string)       // a switch from a page (nuc/ctl)

	mu      sync.Mutex
	clients map[*websocket.Conn]chan []byte
	env     map[string]map[string]any
	reason  []map[string]any
	conv    []map[string]any
	state   map[string]any
	ln      net.Listener
}

func New() *Server {
	return &Server{clients: map[*websocket.Conn]chan []byte{}, env: map[string]map[string]any{},
		state: map[string]any{}}
}

func ts(t time.Time) float64 { return float64(t.UnixNano()) / 1e9 }

// Env is an entity's line (nil: gone).
func (s *Server) Env(key string, text *string, t time.Time) {
	f := map[string]any{"ev": "nuc/env", "key": key, "text": text, "ts": ts(t)}
	s.mu.Lock()
	if text == nil {
		delete(s.env, key)
	} else {
		s.env[key] = f
	}
	s.mu.Unlock()
	s.send(f)
}

// Reasoning is a conclusion of the rule engine.
func (s *Server) Reasoning(text string, t time.Time) {
	f := map[string]any{"ev": "nuc/reasoning", "text": text, "ts": ts(t)}
	s.mu.Lock()
	s.reason = keep(append(s.reason, f), 200)
	s.mu.Unlock()
	s.send(f)
}

// Tool is a tool call and the rules' answer.
func (s *Server) Tool(name, args, answer string) {
	f := map[string]any{"ev": "nuc/tool", "name": name, "args": args, "answer": answer, "ts": ts(time.Now())}
	s.mu.Lock()
	s.reason = keep(append(s.reason, f), 200)
	s.mu.Unlock()
	s.send(f)
}

// Said is a line of the conversation: who heard | asleep | reply (id: a typed turn's).
func (s *Server) Said(who, id, text string) {
	f := map[string]any{"ev": "nuc/" + who, "text": text, "ts": ts(time.Now())}
	if id != "" {
		f["id"] = id
	}
	s.mu.Lock()
	s.conv = keep(append(s.conv, f), 200)
	s.mu.Unlock()
	s.send(f)
}

// Event is one dialog event of the engine, for the page's floor machine and its events list.
func (s *Server) Event(kind string, fields map[string]any) {
	f := map[string]any{"ev": "nuc/event", "kind": kind, "t": float64(time.Now().UnixMilli())}
	for k, v := range fields {
		f[k] = v
	}
	if kind == "fsttm" {
		s.mu.Lock()
		s.state["floor"] = fields["dst"]
		s.mu.Unlock()
	}
	s.send(f)
}

// State is the engine's floor, attention and dialog state (any of them).
func (s *Server) State(kv map[string]any) {
	s.mu.Lock()
	for k, v := range kv {
		s.state[k] = v
	}
	f := map[string]any{"ev": "nuc/state"}
	for k, v := range s.state {
		f[k] = v
	}
	s.mu.Unlock()
	s.send(f)
}

func keep(xs []map[string]any, n int) []map[string]any {
	if len(xs) > n {
		return xs[len(xs)-n:]
	}
	return xs
}

func (s *Server) send(f map[string]any) {
	b, _ := json.Marshal(f)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ch := range s.clients {
		select {
		case ch <- b:
		default: // a slow page loses frames rather than slowing the engine
		}
	}
}

func (s *Server) snapshot() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	env := make([]map[string]any, 0, len(s.env))
	for _, f := range s.env {
		env = append(env, f)
	}
	st := map[string]any{}
	for k, v := range s.state {
		st[k] = v
	}
	b, _ := json.Marshal(map[string]any{"ev": "nuc/snapshot", "env": env, "reasoning": s.reason,
		"conversation": s.conv, "state": st})
	return b
}

func (s *Server) ws(c *websocket.Conn) {
	ch := make(chan []byte, 256)
	s.mu.Lock()
	s.clients[c] = ch
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.clients, c)
		s.mu.Unlock()
		c.Close()
	}()
	if err := websocket.Message.Send(c, string(s.snapshot())); err != nil {
		return
	}
	go func() {
		for {
			var msg string
			if err := websocket.Message.Receive(c, &msg); err != nil {
				c.Close()
				return
			}
			var f struct {
				Ev, ID, Text, Op string
			}
			if json.Unmarshal([]byte(msg), &f) != nil {
				continue
			}
			switch {
			case f.Ev == "nuc/say" && f.Text != "" && s.OnSay != nil:
				s.OnSay(f.ID, f.Text)
			case f.Ev == "nuc/ctl" && f.Op != "" && s.OnCtl != nil:
				s.OnCtl(f.Op)
			}
		}
	}()
	ping := time.NewTicker(25 * time.Second)
	defer ping.Stop()
	for {
		select {
		case b := <-ch:
			if err := websocket.Message.Send(c, string(b)); err != nil {
				return
			}
		case <-ping.C:
			if err := websocket.Message.Send(c, `{"ev":"nuc/ping"}`); err != nil {
				return
			}
		}
	}
}

// Handler is the page and its socket (/ws).
func (s *Server) Handler() http.Handler {
	sub, _ := fs.Sub(webFS, "web")
	mux := http.NewServeMux()
	mux.Handle("/ws", websocket.Server{Handler: s.ws, Handshake: func(*websocket.Config, *http.Request) error { return nil }})
	fileSrv := http.FileServer(http.FS(sub))
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		fileSrv.ServeHTTP(w, r)
	}))
	return mux
}

// Serve listens on addr (0.0.0.0:8767 on the NUC: the LAN and the tailnet).
func (s *Server) Serve(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	s.ln = ln
	log.Printf("[panel] http://%s/", ln.Addr())
	go func() { _ = http.Serve(ln, s.Handler()) }()
	return nil
}

func (s *Server) Addr() string { return s.ln.Addr().String() }
