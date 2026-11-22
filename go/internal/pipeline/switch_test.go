package pipeline

import (
	"strings"
	"testing"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/vad"
	"github.com/ibnHatab/fsttm/go/pkg/fsm"
)

func newSitEngine(spk *fakeSpeaker) *Engine {
	return New(Config{SystemPrompt: "p", GBNF: "g"},
		fakeLLM{json: `{"intent":"LOCAL_ACTION","action":"SIT_DOWN"}`, voice: "Sitting down."},
		fakeSTT{}, spk)
}

// The mic off: what the VAD brings is dropped; on again: heard.
func TestMicOffIgnoresSpeech(t *testing.T) {
	spk := newFakeSpeaker(20 * time.Millisecond)
	e := newSitEngine(spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()
	e.Control("mic off")
	events <- vad.Event{SpeechStart: true}
	events <- utter("sit down")
	time.Sleep(100 * time.Millisecond)
	if len(spk.said()) != 0 || e.Mic() {
		t.Fatalf("mic off: spoke %v", spk.said())
	}
	// switched off in the middle of an utterance: the floor it took comes back
	e.Control("mic on")
	events <- vad.Event{SpeechStart: true}
	waitFor(t, func() bool { return e.Turn.State == fsm.User }, "the floor taken")
	e.Control("mic off")
	events <- utter("half a sentence")
	waitFor(t, func() bool { return e.Turn.State != fsm.User }, "the floor released after the mic went off")
	e.Control("mic on")
	events <- vad.Event{SpeechStart: true}
	events <- utter("sit down")
	waitFor(t, func() bool { return len(spk.said()) == 1 }, "the reply once the mic is on")
}

// The voice off: the turn runs, nothing is spoken, the floor comes back.
func TestVoiceOffSpeaksNothing(t *testing.T) {
	spk := newFakeSpeaker(20 * time.Millisecond)
	e := newSitEngine(spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()
	e.Control("voice off")
	events <- vad.Event{SpeechStart: true}
	events <- utter("sit down")
	waitFor(t, func() bool { return e.Turn.State == fsm.FreeS }, "the floor released")
	if len(spk.said()) != 0 || e.Voice() {
		t.Fatalf("voice off: spoke %v", spk.said())
	}
}

// Stop (and the voice switched off) during a reply cuts the narration at once.
func TestStopHushesNarration(t *testing.T) {
	for _, op := range []string{"stop", "voice off"} {
		spk := newFakeSpeaker(5 * time.Second)
		e := newSitEngine(spk)
		events, cancel, wg := run(t, e)
		events <- vad.Event{SpeechStart: true}
		events <- utter("sit down")
		waitFor(t, func() bool { return len(spk.said()) == 1 }, "speaking")
		t0 := time.Now()
		e.Control(op)
		waitFor(t, func() bool { return e.Turn.State == fsm.FreeS }, op+": the floor released")
		if d := time.Since(t0); d > time.Second {
			t.Fatalf("%s took %v", op, d)
		}
		cancel()
		wg.Wait()
	}
}

// A note's transcript is cut at 160 characters.
func TestShort(t *testing.T) {
	long := strings.Repeat("and without the stereo channel, ", 20)
	s := Short(long)
	if len([]rune(s)) > 162 || !strings.HasSuffix(s, " …") {
		t.Fatalf("Short: %q", s)
	}
	if Short("turn it off") != "turn it off" {
		t.Fatal("a short line changed")
	}
}
