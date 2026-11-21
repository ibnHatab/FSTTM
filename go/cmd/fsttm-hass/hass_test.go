package main

import (
	"bufio"
	"encoding/json"
	"net"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	"github.com/ibnHatab/fsttm/go/pkg/streams"
)

func st(js string) State {
	var s State
	if err := json.Unmarshal([]byte(js), &s); err != nil {
		panic(err)
	}
	return s
}

// tests/test_feeds_hass.py, ported
func TestFilter(t *testing.T) {
	c := Defaults()
	for id, want := range map[string]bool{"fan.lab_supply_fan": true, "sensor.sun_next_dawn": false,
		"number.lab_display_brightness": false, "person.nina": false, "light.lab_display_brightness": false} {
		if got := c.Wanted(id); got != want {
			t.Errorf("%s: %v", id, got)
		}
	}
}

func TestWords(t *testing.T) {
	c := Defaults()
	for js, want := range map[string]string{
		`{"entity_id": "fan.lab_supply_fan", "state": "on", "attributes": {"friendly_name": "lab Supply fan", "percentage": 40, "preset_mode": "night", "preset_modes": ["night", "boost"]}}`: "lab Supply fan: on (percentage 40%, preset mode night)",
		`{"entity_id": "sensor.lab_outside_temperature", "state": "27.8", "attributes": {"friendly_name": "lab Outside temperature", "unit_of_measurement": "°C"}}`:                           "lab Outside temperature: 27.8 °C",
		`{"entity_id": "light.desk", "state": "on", "attributes": {"brightness": 128}}`:                                                                                                       "light.desk: on (brightness 50%)",
		// beyond the Python tests: Python's str() of the values, and the unit kept off "unavailable"
		`{"entity_id": "climate.lab", "state": "heat", "attributes": {"temperature": 21.0, "current_temperature": 19.5, "hvac_action": "heating"}}`: "climate.lab: heat (temperature 21.0, current temperature 19.5, hvac action heating)",
		`{"entity_id": "sensor.t", "state": "unavailable", "attributes": {"unit_of_measurement": "°C"}}`:                                            "sensor.t: unavailable",
		`{"entity_id": "media_player.tv", "state": "off", "attributes": {"source": "", "volume_level": 0, "media_title": null}}`:                    "media_player.tv: off (volume level 0)",
	} {
		if got := c.Words(st(js)); got != want {
			t.Errorf("got  %q\nwant %q", got, want)
		}
	}
}

type sink struct{ evs []streams.Event }

func (s *sink) Send(evs ...streams.Event) bool { s.evs = append(s.evs, evs...); return true }

func TestChanges(t *testing.T) {
	f := NewFeed(Defaults())
	d := &sink{}
	f.Door = d
	n := f.All([]State{st(`{"entity_id": "fan.a", "state": "on", "attributes": {}}`), st(`{"entity_id": "person.x", "state": "home"}`)})
	if n != 1 || len(d.evs) != 1 || *d.evs[0].Text != "fan.a: on" {
		t.Fatalf("all: %d %+v", n, d.evs)
	}
	s2 := st(`{"entity_id": "fan.a", "state": "on", "attributes": {"preset_modes": ["x"]}}`)
	if f.Changed("fan.a", &s2) {
		t.Error("attribute churn sent")
	}
	s3 := st(`{"entity_id": "fan.a", "state": "off", "attributes": {}}`)
	if !f.Changed("fan.a", &s3) || *d.evs[1].Text != "fan.a: off" {
		t.Error("change not sent")
	}
	if !f.Changed("fan.a", nil) || d.evs[2].Text != nil {
		t.Error("removal not sent as null")
	}
	if len(f.Known()) != 0 {
		t.Error("a removed entity in the resend")
	}
}

// TestSession: a fake Home Assistant and a fake door; the door restarts and gets the whole
// state again on reconnect.
func TestSession(t *testing.T) {
	events := make(chan string, 4)
	ha := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
		websocket.JSON.Send(ws, map[string]string{"type": "auth_required"})
		var m map[string]any
		websocket.JSON.Receive(ws, &m)
		if m["access_token"] != "tok" {
			websocket.JSON.Send(ws, map[string]string{"type": "auth_invalid", "message": "bad"})
			return
		}
		websocket.JSON.Send(ws, map[string]string{"type": "auth_ok", "ha_version": "test"})
		websocket.JSON.Receive(ws, &m) // get_states
		websocket.JSON.Receive(ws, &m) // subscribe
		websocket.JSON.Send(ws, map[string]any{"id": 1, "type": "result", "success": true, "result": []any{
			map[string]any{"entity_id": "light.desk", "state": "on", "attributes": map[string]any{"brightness": 255}},
			map[string]any{"entity_id": "sun.sun", "state": "above_horizon"},
		}})
		for e := range events {
			websocket.JSON.Send(ws, map[string]any{"id": 2, "type": "event", "event": map[string]any{"data": map[string]any{
				"entity_id": "light.desk", "new_state": map[string]any{"entity_id": "light.desk", "state": e}}}})
		}
	}))
	defer ha.Close()
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	lines := make(chan string, 16)
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				sc := bufio.NewScanner(c)
				for sc.Scan() {
					lines <- sc.Text()
				}
			}(c)
		}
	}()
	f := NewFeed(Defaults())
	door := &streams.Client{Addr: ln.Addr().String(), OnConnect: f.Known}
	f.Door = door
	url := "ws" + strings.TrimPrefix(ha.URL, "http")
	if err := f.Follow(url, "bad"); err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("bad token: %v", err)
	}
	go f.Follow(url, "tok")
	prev := ""
	next := func() string { // the next line that differs from the one before (a fresh
		// connection's OnConnect state may repeat the batch it opens with)
		for {
			select {
			case l := <-lines:
				if l != prev {
					prev = l
					return l
				}
			case <-time.After(3 * time.Second):
				t.Fatal("no line")
				return ""
			}
		}
	}
	first := next() // the first connection: OnConnect (empty then) + the batch
	if !strings.Contains(first, `"key":"light.desk"`) || !strings.Contains(first, "brightness 100%") {
		t.Fatalf("first: %s", first)
	}
	events <- "off"
	if l := next(); !strings.Contains(l, `"text":"light.desk: off"`) {
		t.Fatalf("change: %s", l)
	}
	door.Close() // the engine restarted: the next connection starts with the whole state
	f.mu.Lock()
	f.last["fan.other"] = new(string)
	*f.last["fan.other"] = "fan.other: on"
	f.mu.Unlock()
	events <- "on"
	got := map[string]bool{}
	for len(got) < 2 {
		l := next()
		got[l] = true
	}
	if !got[`{"stream":"env","key":"fan.other","text":"fan.other: on"}`] || !got[`{"stream":"env","key":"light.desk","text":"light.desk: on"}`] {
		t.Fatalf("resend: %v", got)
	}
	close(events)
}
