package streams

import (
	"sync"
	"testing"
	"time"
)

func TestDoorAndClient(t *testing.T) {
	var mu sync.Mutex
	var got []Event
	d, err := Open("127.0.0.1:0", func(e Event) { mu.Lock(); got = append(got, e); mu.Unlock() })
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	on := "lab fan: on"
	cl := &Client{Addr: d.Addr(), OnConnect: func() []Event { return []Event{EnvLine("sensor.t", &on)} }}
	defer cl.Close()
	if !cl.Send(ReasoningLine("air done: I start the lab ventilation."), EnvLine("fan.lab", nil)) {
		t.Fatal("send failed")
	}
	for i := 0; i < 100; i++ {
		mu.Lock()
		n := len(got)
		mu.Unlock()
		if n == 3 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 3 || got[0].Key != "sensor.t" || got[1].Stream != "reasoning" || got[2].Text != nil {
		t.Fatalf("%+v", got)
	}
}

func TestParse(t *testing.T) {
	if _, err := Parse([]byte(`{"stream": "env", "text": "x"}`)); err == nil {
		t.Fatal("env without a key")
	}
	if _, err := Parse([]byte(`{"stream": "nope"}`)); err == nil {
		t.Fatal("unknown stream")
	}
	e, err := Parse([]byte(`{"stream": "reasoning", "text": "x", "ts": 1.5}`))
	if err != nil || e.TS.UnixMilli() != 1500 {
		t.Fatalf("%v %v", e, err)
	}
}
