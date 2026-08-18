package pipeline

// End-to-end behavior of the orchestrator with fake drivers: FSM traces per
// N09-1071 phenomena, barge-in cutting the speaker (transition 8), half-
// duplex echo suppression, and spoken-output selection. No models, no audio.

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/fsm"
	"github.com/ibnHatab/fsttm/go/internal/llm"
	"github.com/ibnHatab/fsttm/go/internal/stt"
	"github.com/ibnHatab/fsttm/go/internal/tts"
	"github.com/ibnHatab/fsttm/go/internal/vad"
)

// ── fakes ─────────────────────────────────────────────────────────────────────

type fakeSTT struct{}

func (fakeSTT) Transcribe(pcm []byte) (stt.Result, bool) {
	return stt.Result{Text: string(pcm)}, true // pcm bytes ARE the text
}

type fakeLLM struct{ json, voice string }

func (f fakeLLM) TwoPass(_, _, _ string) (*llm.Result, error) {
	return &llm.Result{JSON: f.json, Voice: f.voice}, nil
}

// fakeSpeaker blocks for `dur` unless cancelled; records everything spoken.
type fakeSpeaker struct {
	mu       sync.Mutex
	spoken   []string
	dur      time.Duration
	cancelCh chan struct{}
}

func newFakeSpeaker(dur time.Duration) *fakeSpeaker {
	return &fakeSpeaker{dur: dur, cancelCh: make(chan struct{}, 4)}
}

func (f *fakeSpeaker) Speak(ctx context.Context, text string) (tts.SpeakResult, error) {
	f.mu.Lock()
	f.spoken = append(f.spoken, text)
	f.mu.Unlock()
	select {
	case <-time.After(f.dur):
		return tts.SpeakResult{Completed: true, Played: f.dur, Synthesized: f.dur}, nil
	case <-f.cancelCh:
		return tts.SpeakResult{Completed: false, Played: f.dur / 3,
			Synthesized: f.dur}, nil
	case <-ctx.Done():
		return tts.SpeakResult{}, ctx.Err()
	}
}

func (f *fakeSpeaker) Cancel() { f.cancelCh <- struct{}{} }

func (f *fakeSpeaker) said() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.spoken...)
}

// trace records every FSM transition the engine makes.
func trace(e *Engine) *[]string {
	log := &[]string{}
	var mu sync.Mutex
	e.Turn.OnChange = func(event string, src, dst fsm.State) {
		mu.Lock()
		*log = append(*log, string(src)+"-("+event+")->"+string(dst))
		mu.Unlock()
	}
	return log
}

func run(t *testing.T, e *Engine) (chan vad.Event, context.CancelFunc, *sync.WaitGroup) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	events := make(chan vad.Event, 16)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); e.Run(ctx, events, nil) }()
	return events, cancel, &wg
}

func utter(text string) vad.Event { return vad.Event{Utterance: []byte(text)} }

func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	for i := 0; i < 200; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for %s", what)
}

// ── tests ─────────────────────────────────────────────────────────────────────

// A full voice turn walks the paper's gap-transition cycle:
// USER →(W,R) FREEu →(G,W) SYSTEM →(R,W) FREEs.
func TestFullTurnFSMTrace(t *testing.T) {
	spk := newFakeSpeaker(50 * time.Millisecond)
	e := New(Config{SystemPrompt: "p", GBNF: "g"},
		fakeLLM{json: `{"intent":"LOCAL_ACTION","action":"SIT_DOWN"}`,
			voice: "Sitting down."},
		fakeSTT{}, spk)
	tr := trace(e)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- vad.Event{SpeechStart: true}
	events <- utter("sit down")
	waitFor(t, func() bool { return e.Turn.State == fsm.FreeS }, "FREEs")

	got := strings.Join(*tr, " ")
	for _, want := range []string{
		"USER-(W_R)->FREEu",   // user releases (transition 4)
		"FREEu-(G_W)->SYSTEM", // system grabs after gap (transition 5)
		"SYSTEM-(R_W)->FREEs", // system releases at end of prompt (transition 1)
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %s in trace: %s", want, got)
		}
	}
	if said := spk.said(); len(said) != 1 || said[0] != "Sitting down." {
		t.Fatalf("spoken = %v", said)
	}
}

// Barge-in (BargeIn=true): a VAD onset after the grace period cuts the
// speaker and the floor walks SYSTEM →(K,G) BOTHs →(R,K) USER (transitions
// 7+8 — the paper's successful barge-in).
func TestBargeInCutsSpeakerAndYieldsFloor(t *testing.T) {
	spk := newFakeSpeaker(5 * time.Second) // long narration to interrupt
	e := New(Config{SystemPrompt: "p", GBNF: "g", BargeIn: true},
		fakeLLM{json: `{"intent":"LOCAL_ACTION","action":"STAND_UP"}`,
			voice: "Standing up now, this takes a while."},
		fakeSTT{}, spk)
	tr := trace(e)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- vad.Event{SpeechStart: true}
	events <- utter("stand up")
	waitFor(t, func() bool { return e.Turn.State == fsm.System }, "SYSTEM (narrating)")

	time.Sleep(bargeGrace + 50*time.Millisecond) // ride out the grace window
	events <- vad.Event{SpeechStart: true}       // the barge-in
	waitFor(t, func() bool { return e.Turn.IsUser() }, "floor back to user")

	got := strings.Join(*tr, " ")
	if !strings.Contains(got, "SYSTEM-(K_G)->BOTHs") ||
		!strings.Contains(got, "BOTHs-(R_K)->USER") {
		t.Fatalf("barge-in must walk SYSTEM→BOTHs→USER, got: %s", got)
	}
}

// Grace window: a VAD blip right after narration starts must NOT cut it
// (TTS onset transient / leftover frames from the user's own utterance).
func TestBargeInGraceSuppresssesEarlyBlip(t *testing.T) {
	spk := newFakeSpeaker(900 * time.Millisecond)
	e := New(Config{SystemPrompt: "p", GBNF: "g", BargeIn: true},
		fakeLLM{json: `{"intent":"STOP"}`, voice: "Stopping."},
		fakeSTT{}, spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- utter("stop")
	waitFor(t, func() bool { return e.Turn.State == fsm.System }, "SYSTEM")
	events <- vad.Event{SpeechStart: true} // within the 600 ms grace
	time.Sleep(150 * time.Millisecond)
	if !e.Turn.IsSystem() {
		t.Fatal("early blip must not cut the narration")
	}
}

// Half-duplex (BargeIn=false): utterances whose audio overlaps our own
// playback are echo and must be dropped — no second LLM turn, no self-loop.
func TestHalfDuplexDropsEcho(t *testing.T) {
	spk := newFakeSpeaker(300 * time.Millisecond)
	llmCalls := 0
	var mu sync.Mutex
	countingLLM := llmFunc(func() (*llm.Result, error) {
		mu.Lock()
		llmCalls++
		mu.Unlock()
		return &llm.Result{JSON: `{"intent":"STOP"}`, Voice: "ok"}, nil
	})
	e := New(Config{SystemPrompt: "p", GBNF: "g"}, countingLLM, fakeSTT{}, spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- utter("stop")
	waitFor(t, func() bool { return e.Turn.State == fsm.System }, "SYSTEM")
	// echo: an utterance that STARTED while we were speaking (its length
	// covers the whole playback window)
	events <- utter(strings.Repeat("x", 32000)) // 1 s of fake PCM
	waitFor(t, func() bool { return e.Turn.State == fsm.FreeS }, "FREEs")
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if llmCalls != 1 {
		t.Fatalf("echo reached the LLM: %d calls", llmCalls)
	}
}

type llmFunc func() (*llm.Result, error)

func (f llmFunc) TwoPass(_, _, _ string) (*llm.Result, error) { return f() }

// Output behavior: deterministic QUERY answers beat the LLM ack; JSON
// parroted as a voice ack is never spoken.
func TestOutputSelection(t *testing.T) {
	spk := newFakeSpeaker(10 * time.Millisecond)
	e := New(Config{SystemPrompt: "p", GBNF: "g"},
		fakeLLM{json: `{"intent":"QUERY","target":{"type":"OBJECT","description":"chair"}}`,
			voice: "ignored"},
		fakeSTT{}, spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- utter("where is the chair")
	waitFor(t, func() bool { return len(spk.said()) == 1 }, "spoken answer")
	if said := spk.said()[0]; said != "I don't see chair in my map yet." {
		t.Fatalf("QUERY must be answered from the map, got %q", said)
	}
}

func TestJSONParrotNeverSpoken(t *testing.T) {
	spk := newFakeSpeaker(10 * time.Millisecond)
	e := New(Config{SystemPrompt: "p", GBNF: "g"},
		fakeLLM{json: `{"intent":"FIND","target":{"type":"OBJECT","description":"ball"}}`,
			voice: `{"intent":"FIND","target":{"type":"OBJECT"...`},
		fakeSTT{}, spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- utter("find the ball")
	waitFor(t, func() bool { return len(spk.said()) == 1 }, "spoken answer")
	if said := spk.said()[0]; strings.ContainsAny(said, "{}") {
		t.Fatalf("spoke raw JSON: %q", said)
	}
}

// Wake word: asleep engine ignores commands, wakes on the word, strips it.
func TestWakeWord(t *testing.T) {
	spk := newFakeSpeaker(10 * time.Millisecond)
	e := New(Config{SystemPrompt: "p", GBNF: "g", WakeWord: "rex"},
		fakeLLM{json: `{"intent":"LOCAL_ACTION","action":"SIT_DOWN"}`,
			voice: "Sitting."},
		fakeSTT{}, spk)
	events, cancel, wg := run(t, e)
	defer func() { cancel(); wg.Wait() }()

	events <- utter("sit down")                  // asleep → ignored
	time.Sleep(100 * time.Millisecond)
	if len(spk.said()) != 0 {
		t.Fatal("asleep engine must not act")
	}
	events <- utter("hey rex, sit down")         // wake + trailing command
	waitFor(t, func() bool { return len(spk.said()) == 1 }, "spoken after wake")
}
