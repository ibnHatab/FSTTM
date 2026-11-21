package attention

import (
	"testing"
	"time"
)

func TestWakeVariantsAndRemainder(t *testing.T) {
	a := New(true, []string{"nina", "hey nina", "hello nina"}, nil, true)
	cases := map[string]struct {
		wake bool
		rem  string
	}{
		"Hello Nina!":                    {true, ""},
		"hey nina, go to the kitchen":    {true, "go to the kitchen"},
		"Nina go to the kitchen":         {true, "go to the kitchen"},
		"go to the kitchen nina":         {true, "go to the kitchen"},
		"the banana is yellow":           {false, ""}, // no substring false-hit
		"go to the kitchen":              {false, ""},
	}
	for text, want := range cases {
		got, rem := a.MatchWake(text)
		if got != want.wake || rem != want.rem {
			t.Errorf("MatchWake(%q) = (%v,%q), want (%v,%q)",
				text, got, rem, want.wake, want.rem)
		}
	}
}

func TestSleepRequiresWakePrefix(t *testing.T) {
	a := New(true, []string{"nina", "hey nina"}, nil, false) // start awake
	// bare sleep phrase (maybe an STT-garbled command) must NOT sleep
	d := a.OnUtterance("go to sleep")
	if d.Action != "command" || a.State != Awake {
		t.Fatalf("bare sleep phrase must stay a command, got %+v", d)
	}
	// addressed by name → sleeps
	d = a.OnUtterance("nina, go to sleep")
	if d.Action != "sleep" || a.State != Asleep {
		t.Fatalf("prefixed sleep must sleep, got %+v state=%s", d, a.State)
	}
	// asleep: commands ignored, wake word wakes with trailing command
	if d := a.OnUtterance("stand up"); d.Action != "ignore" {
		t.Fatalf("asleep must ignore, got %+v", d)
	}
	if d := a.OnUtterance("hey nina stand up"); d.Action != "wake" ||
		d.Text != "stand up" {
		t.Fatalf("wake with command, got %+v", d)
	}
}

func TestDisabledIsAlwaysCommand(t *testing.T) {
	a := New(false, nil, nil, true)
	if d := a.OnUtterance("anything at all"); d.Action != "command" {
		t.Fatalf("disabled layer must pass commands, got %+v", d)
	}
}

func TestAwakeForFallsBackAsleep(t *testing.T) {
	now := time.Unix(0, 0)
	a := New(true, []string{"hey nina"}, nil, true)
	a.AwakeFor = 30 * time.Second
	a.Now = func() time.Time { return now }
	if d := a.OnUtterance("hey nina turn on the light"); d.Action != "wake" {
		t.Fatalf("%+v", d)
	}
	now = now.Add(20 * time.Second)
	if d := a.OnUtterance("and the tv"); d.Action != "command" {
		t.Fatalf("%+v", d)
	}
	now = now.Add(25 * time.Second)
	a.Touch() // a reply ended
	now = now.Add(25 * time.Second)
	if !a.Awake() {
		t.Fatal("asleep 25 s after the reply")
	}
	now = now.Add(6 * time.Second)
	if a.Awake() || a.OnUtterance("what about the tv").Action != "ignore" {
		t.Fatal("still awake after 31 s quiet")
	}
}
