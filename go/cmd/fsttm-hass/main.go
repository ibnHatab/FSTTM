// fsttm-hass: Home Assistant → the ENVIRONMENT stream (the Go port of fsttm/feeds/hass.py).
//
// It connects to Home Assistant's websocket API, reads every state once, then follows
// state_changed. Each entity that passes the filter goes to the engine's stream door as
// {"stream": "env", "key": <entity_id>, "text": <its state in words>}; a removed entity
// goes with text null. It reconnects to both forever (Home Assistant restarts, the NUC
// boots in any order), and a new door connection first carries the whole current state,
// so an engine that restarted is current at once.
//
//	fsttm-hass -config deploy/nuc/fsttm-go.yaml     (its hass: block, streams.addr)
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/websocket"
	"gopkg.in/yaml.v3"

	"github.com/ibnHatab/fsttm/go/pkg/streams"
)

type fileConfig struct {
	Hass    Config `yaml:"hass"`
	Streams struct {
		Addr string `yaml:"addr"`
		Port int    `yaml:"port"` // the Python config's form
	} `yaml:"streams"`
}

// Feed keeps the words last sent per entity and sends the changes.
type Feed struct {
	Cfg  Config
	Door interface{ Send(...streams.Event) bool }
	mu   sync.Mutex
	last map[string]*string
}

func NewFeed(cfg Config) *Feed { return &Feed{Cfg: cfg, last: map[string]*string{}} }

// Known is the current state as events (the door's OnConnect), sorted by entity.
func (f *Feed) Known() []streams.Event {
	f.mu.Lock()
	defer f.mu.Unlock()
	keys := make([]string, 0, len(f.last))
	for k, t := range f.last {
		if t != nil {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	evs := make([]streams.Event, len(keys))
	for i, k := range keys {
		evs[i] = streams.EnvLine(k, f.last[k])
	}
	return evs
}

// All is get_states' result: every wanted entity, sent at once.
func (f *Feed) All(states []State) int {
	var evs []streams.Event
	f.mu.Lock()
	for _, st := range states {
		if !f.Cfg.Wanted(st.EntityID) {
			continue
		}
		t := f.Cfg.Words(st)
		f.last[st.EntityID] = &t
		evs = append(evs, streams.EnvLine(st.EntityID, &t))
	}
	f.mu.Unlock()
	if f.Door != nil {
		f.Door.Send(evs...)
	}
	return len(evs)
}

// Changed is one state_changed; false when it is filtered out or its words did not change
// (attribute churn).
func (f *Feed) Changed(entityID string, newState *State) bool {
	if !f.Cfg.Wanted(entityID) {
		return false
	}
	var t *string
	if newState != nil {
		w := f.Cfg.Words(*newState)
		t = &w
	}
	f.mu.Lock()
	old, seen := f.last[entityID]
	if seen && ((old == nil && t == nil) || (old != nil && t != nil && *old == *t)) {
		f.mu.Unlock()
		return false
	}
	f.last[entityID] = t
	f.mu.Unlock()
	if f.Door != nil {
		f.Door.Send(streams.EnvLine(entityID, t))
	}
	return true
}

var errAuth = errors.New("Home Assistant refused the token")

type haMsg struct {
	ID        int             `json:"id"`
	Type      string          `json:"type"`
	Message   string          `json:"message"`
	HAVersion string          `json:"ha_version"`
	Result    json.RawMessage `json:"result"`
	Event     struct {
		Data struct {
			EntityID string `json:"entity_id"`
			NewState *State `json:"new_state"`
		} `json:"data"`
	} `json:"event"`
}

// Follow is one websocket session: auth, get_states, subscribe, then the events.
func (f *Feed) Follow(url, token string) error {
	origin := "http://localhost/"
	ws, err := websocket.Dial(url, "", origin)
	if err != nil {
		return err
	}
	defer ws.Close()
	ws.MaxPayloadBytes = 1 << 24
	recv := func() (haMsg, error) {
		var m haMsg
		err := websocket.JSON.Receive(ws, &m)
		return m, err
	}
	hello, err := recv()
	if err != nil {
		return err
	}
	if hello.Type != "auth_required" {
		return fmt.Errorf("unexpected greeting: %s", hello.Type)
	}
	if err := websocket.JSON.Send(ws, map[string]string{"type": "auth", "access_token": token}); err != nil {
		return err
	}
	ok, err := recv()
	if err != nil {
		return err
	}
	if ok.Type != "auth_ok" {
		return fmt.Errorf("%w: %s", errAuth, ok.Message)
	}
	if err := websocket.JSON.Send(ws, map[string]any{"id": 1, "type": "get_states"}); err != nil {
		return err
	}
	if err := websocket.JSON.Send(ws, map[string]any{"id": 2, "type": "subscribe_events", "event_type": "state_changed"}); err != nil {
		return err
	}
	log.Printf("connected to %s (Home Assistant %s)", url, ok.HAVersion)
	for {
		m, err := recv()
		if err != nil {
			return err
		}
		switch {
		case m.ID == 1 && m.Type == "result":
			var states []State
			_ = json.Unmarshal(m.Result, &states)
			log.Printf("sent %d entities", f.All(states))
		case m.Type == "event":
			d := m.Event.Data
			if f.Changed(d.EntityID, d.NewState) {
				t := "None"
				if d.NewState != nil {
					t = f.Cfg.Words(*d.NewState)
				}
				log.Printf("%s -> %s", d.EntityID, t)
			}
		}
	}
}

func expand(p string) string {
	if strings.HasPrefix(p, "~/") {
		h, _ := os.UserHomeDir()
		return filepath.Join(h, p[2:])
	}
	return p
}

func main() {
	cfgPath := flag.String("config", "deploy/nuc/fsttm-go.yaml", "YAML config (its hass: and streams: blocks)")
	flag.Parse()
	log.SetFlags(log.LstdFlags)
	raw, err := os.ReadFile(*cfgPath)
	if err != nil {
		log.Fatal(err)
	}
	fc := fileConfig{Hass: Defaults()}
	if err := yaml.Unmarshal(raw, &fc); err != nil {
		log.Fatal(err)
	}
	addr := fc.Streams.Addr
	if addr == "" {
		port := fc.Streams.Port
		if port == 0 {
			port = 8766
		}
		addr = fmt.Sprintf("127.0.0.1:%d", port)
	}
	if !fc.Hass.Dumps() {
		log.Printf("hass.dump is false: metis-home's semantic world serves ENVIRONMENT; idle (set dump: true for the raw entity dump)")
		for { // idle (no goroutine to wait on: select{} would be a deadlock)
			time.Sleep(time.Hour)
		}
	}
	tok, err := os.ReadFile(expand(fc.Hass.TokenFile))
	if err != nil {
		log.Fatalf("the Home Assistant token: %v", err)
	}
	token := strings.TrimSpace(string(tok))
	feed := NewFeed(fc.Hass)
	door := &streams.Client{Addr: addr, OnConnect: feed.Known}
	feed.Door = door
	go func() { // while the door is closed (the engine not up, or restarting), knock
		for range time.Tick(5 * time.Second) {
			if len(feed.Known()) > 0 {
				door.Send()
			}
		}
	}()
	wait := time.Second
	for {
		t0 := time.Now()
		err := feed.Follow(fc.Hass.URL, token)
		if errors.Is(err, errAuth) {
			log.Fatal(err)
		}
		if time.Since(t0) > time.Minute {
			wait = time.Second // it was up: start again quickly
		}
		log.Printf("Home Assistant: %v; again in %.0f s", err, wait.Seconds())
		time.Sleep(wait)
		if wait *= 2; wait > 30*time.Second {
			wait = 30 * time.Second
		}
	}
}
