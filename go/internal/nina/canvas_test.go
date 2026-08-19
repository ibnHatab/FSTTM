package nina

// Fidelity: the Go canvas port must reproduce the Python reference
// (backends_nina.SemanticMemoryNina) on the same pack — testdata generated
// by the reference math with a planted "chair" cluster.

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/ibnHatab/fsttm/go/internal/intent"
)

type refCand struct {
	InstanceID int        `json:"instance_id"`
	Score      float64    `json:"score"`
	Position   [3]float64 `json:"position"`
}

type refFile struct {
	Cells int       `json:"cells_after_zfilter"`
	Chair []refCand `json:"chair"`
}

func TestCanvasMatchesPythonReference(t *testing.T) {
	c, err := LoadCanvas("testdata/pack.npz", "testdata/phrases.npz")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/expected.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref refFile
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if len(c.xyz) != ref.Cells {
		t.Fatalf("z-filter kept %d cells, reference %d", len(c.xyz), ref.Cells)
	}
	got := c.Query(intent.SemanticTarget{Type: "OBJECT",
		Description: "chair"}, nil)
	if len(got) != len(ref.Chair) {
		t.Fatalf("got %d candidates, reference %d", len(got), len(ref.Chair))
	}
	for i, rc := range ref.Chair {
		if math.Abs(got[i].Score-rc.Score) > 1e-4 {
			t.Fatalf("cand %d score %.6f vs reference %.6f",
				i, got[i].Score, rc.Score)
		}
		for j := 0; j < 3; j++ {
			if math.Abs(got[i].Position[j]-rc.Position[j]) > 1e-4 {
				t.Fatalf("cand %d pos[%d] %.5f vs %.5f",
					i, j, got[i].Position[j], rc.Position[j])
			}
		}
	}
}

func TestCanvasUnknownPhraseFindsNothing(t *testing.T) {
	c, err := LoadCanvas("testdata/pack.npz", "testdata/phrases.npz")
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Query(intent.SemanticTarget{Description: "unicorn"}, nil); got != nil {
		t.Fatalf("unknown phrase must return nil (no tower), got %v", got)
	}
}

// ── link + backends contract ─────────────────────────────────────────────────

func TestLinkEmitsContractedEventsOnly(t *testing.T) {
	var buf bytes.Buffer
	link := NewLink(&buf)
	d := NewDispatcher(link, nil)

	c, _ := intent.Parse(`{"intent":"LOCAL_ACTION","action":"SIT_DOWN"}`)
	d.Handle(c)
	c, _ = intent.Parse(`{"intent":"STOP"}`)
	d.Handle(c)
	// JUMP is in the LLM enum but NOT voice-safe — must emit nothing
	c, _ = intent.Parse(`{"intent":"LOCAL_ACTION","action":"JUMP"}`)
	d.Handle(c)

	var events []map[string]any
	dec := json.NewDecoder(&buf)
	for dec.More() {
		var ev map[string]any
		if err := dec.Decode(&ev); err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	// SIT_DOWN action name, then STOP twice (actions.Stop + nav.Cancel —
	// both immediate per spec §14). The engine speaks NAMES on
	// /nina/action; api-ids live only in the cmd_arbiter (single motion
	// author — stand-up incident 2026-08-19).
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %v", events)
	}
	if events[0]["ev"] != "action" || events[0]["name"] != "SIT_DOWN" {
		t.Fatalf(`SIT_DOWN → {ev:action name:SIT_DOWN}, got %v`, events[0])
	}
	for _, ev := range events[1:] {
		if ev["ev"] != "action" || ev["name"] != "STOP" {
			t.Fatalf("STOP → action STOP, got %v", ev)
		}
	}
	for _, ev := range events {
		if ev["ev"] == "sport" {
			t.Fatalf("voice stack must NEVER emit a sport event: %v", ev)
		}
	}
}

func TestTurnIsBoundedBurstEndingInZero(t *testing.T) {
	var buf bytes.Buffer
	link := NewLink(&buf)
	a := &Actions{Link: link}
	c, _ := intent.Parse(`{"intent":"LOCAL_ACTION","action":"TURN","direction":"RIGHT","angle_deg":20}`)
	a.Execute(c) // 20° at 0.4 rad/s ≈ 0.87 s → ~9 ticks

	var wz []float64
	dec := json.NewDecoder(&buf)
	for dec.More() {
		var ev struct {
			Ev string  `json:"ev"`
			Wz float64 `json:"wz"`
		}
		if err := dec.Decode(&ev); err != nil {
			t.Fatal(err)
		}
		if ev.Ev != "cmd_vel" {
			t.Fatalf("turn must emit only cmd_vel, got %s", ev.Ev)
		}
		wz = append(wz, ev.Wz)
	}
	if len(wz) < 5 || len(wz) > 15 {
		t.Fatalf("bounded burst expected (~9 ticks), got %d", len(wz))
	}
	for _, v := range wz[:len(wz)-1] {
		if v != -0.4 { // RIGHT = negative, arbiter-shaped constant rate
			t.Fatalf("tick wz = %v, want -0.4", v)
		}
	}
	if wz[len(wz)-1] != 0 {
		t.Fatal("burst must END with a zero Twist")
	}
}

func TestNavigateEmitsMapGoal(t *testing.T) {
	var buf bytes.Buffer
	link := NewLink(&buf)
	n := &Nav{Link: link}
	n.Navigate([3]float64{4.2, 1.7, 0.0}, 17)
	var ev map[string]any
	if err := json.Unmarshal(buf.Bytes(), &ev); err != nil {
		t.Fatal(err)
	}
	if ev["ev"] != "nav_goal" || ev["x"].(float64) != 4.2 {
		t.Fatalf("bad nav_goal event: %v", ev)
	}
}

// ── motion safety ────────────────────────────────────────────────────────────
// The motion-inhibit gate (arm_motion flag) was RETIRED 2026-08-19 by the
// arbiter contract: the engine has no Sport method at all (compile-time
// guarantee) and publishes action NAMES on /nina/action; cmd_arbiter is the
// single motion author and gates on its armed state. See
// TestLinkEmitsContractedEventsOnly for the contract assertions.
