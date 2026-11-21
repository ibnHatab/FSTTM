// Package streams is the protocol of the two streams that run beside the conversation and
// reach the LLM as context, never as turns: ENVIRONMENT (the home's state, keyed by entity,
// a newer line replacing the older) and REASONING (the rule engine's conclusions, appended).
// The engine opens a Door (a local TCP port, 127.0.0.1:8766 by default); producers (the Home
// Assistant feed, the Metis home service) write one JSON object per line with a Client.
// The same protocol as fsttm/streams.py:
//
//	{"stream": "env", "key": "fan.lab_supply_fan", "text": "lab Supply fan: on (percentage 33%)"}
//	{"stream": "env", "key": "...", "text": null}          the entity is gone
//	{"stream": "reasoning", "text": "air done: I start the lab ventilation."}
package streams

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"time"
)

// DefaultAddr is the engine's door.
const DefaultAddr = "127.0.0.1:8766"

// Event is one line.
type Event struct {
	Stream string    `json:"stream"` // "env" | "reasoning"
	Key    string    `json:"key,omitempty"`
	Text   *string   `json:"text"`
	TS     time.Time `json:"-"` // when it came in (or the line's "ts", seconds since the epoch)
}

// Parse checks one line.
func Parse(line []byte) (Event, error) {
	var raw struct {
		Stream string   `json:"stream"`
		Key    string   `json:"key"`
		Text   *string  `json:"text"`
		TS     *float64 `json:"ts"`
	}
	if err := json.Unmarshal(line, &raw); err != nil {
		return Event{}, err
	}
	ev := Event{Stream: raw.Stream, Key: raw.Key, Text: raw.Text, TS: time.Now()}
	if raw.TS != nil && *raw.TS > 0 {
		ev.TS = time.Unix(0, int64(*raw.TS*1e9))
	}
	switch raw.Stream {
	case "env":
		if raw.Key == "" {
			return Event{}, errors.New("an env entry needs a key (the entity)")
		}
	case "reasoning":
		if raw.Text == nil || *raw.Text == "" {
			return Event{}, errors.New("a reasoning entry needs a text")
		}
	default:
		return Event{}, errors.New(`a line is {"stream": "env"|"reasoning", "key": ..., "text": ...}`)
	}
	return ev, nil
}

// Door listens on addr and hands every good line to deliver (from the connection's
// goroutine, so deliver must be safe for concurrent use); a bad line is answered
// "error: ..." on the same connection.
type Door struct {
	ln      net.Listener
	deliver func(Event)
	mu      sync.Mutex
	count   int
}

// Count is how many lines were delivered.
func (d *Door) Count() int { d.mu.Lock(); defer d.mu.Unlock(); return d.count }

func Open(addr string, deliver func(Event)) (*Door, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	d := &Door{ln: ln, deliver: deliver}
	go d.accept()
	log.Printf("[streams] env + reasoning in on %s", ln.Addr())
	return d, nil
}

func (d *Door) Addr() string { return d.ln.Addr().String() }

func (d *Door) Close() error { return d.ln.Close() }

func (d *Door) accept() {
	for {
		c, err := d.ln.Accept()
		if err != nil {
			return
		}
		go func() {
			defer c.Close()
			sc := bufio.NewScanner(c)
			sc.Buffer(make([]byte, 1<<16), 1<<20)
			for sc.Scan() {
				line := sc.Bytes()
				if len(line) == 0 {
					continue
				}
				ev, err := Parse(line)
				if err != nil {
					fmt.Fprintf(c, "error: %v\n", err)
					continue
				}
				d.mu.Lock()
				d.count++
				d.mu.Unlock()
				d.deliver(ev)
			}
		}()
	}
}

// ── the producer's side ─────────────────────────────────────────────────────────────────

// EnvLine is an env entry (text nil: the entity is gone).
func EnvLine(key string, text *string) Event { return Event{Stream: "env", Key: key, Text: text} }

// ReasoningLine is a conclusion of the rule engine.
func ReasoningLine(text string) Event { return Event{Stream: "reasoning", Text: &text} }

// Client writes events to a door, reconnecting when the engine restarts. OnConnect, when
// set, gives the events a new connection starts with (a producer's whole current state,
// so an engine that restarted is current at once).
type Client struct {
	Addr      string
	OnConnect func() []Event
	mu        sync.Mutex
	c         net.Conn
}

// Send writes the events; false when the door is not there (the events are dropped; the
// next connection carries OnConnect's state).
func (cl *Client) Send(evs ...Event) bool {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	for try := 0; try < 2; try++ {
		if cl.c == nil {
			c, err := net.DialTimeout("tcp", cl.Addr, 3*time.Second)
			if err != nil {
				return false
			}
			cl.c = c
			if cl.OnConnect != nil {
				evs = append(cl.OnConnect(), evs...)
			}
		}
		var b []byte
		for _, e := range evs {
			line, _ := json.Marshal(e)
			b = append(append(b, line...), '\n')
		}
		_ = cl.c.SetWriteDeadline(time.Now().Add(5 * time.Second))
		if _, err := cl.c.Write(b); err == nil {
			return true
		}
		cl.c.Close()
		cl.c = nil
	}
	return false
}

func (cl *Client) Close() {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	if cl.c != nil {
		cl.c.Close()
		cl.c = nil
	}
}
