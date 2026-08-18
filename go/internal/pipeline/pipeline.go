// Package pipeline wires the engine with message passing: every stage is a
// goroutine, every hand-off a channel, and the orchestrator goroutine is the
// SINGLE owner of the turn-taking FSM (no locks, no shared mutable state).
//
//	capture ──frames──▶ vad ──events──▶ orchestrator ──speak──▶ tts
//	                                       │  ▲                   │
//	                                       ▼  └────ttsDone────────┘
//	                                   stt / llm (called inline,
//	                                   bursty; idle between turns)
//
// Idle behaviour — the whole point of the Go port: every goroutine blocks on
// a channel receive. With a silent mic the only work is the 20 ms WebRTC-VAD
// call (native, microseconds); with no audio at all the process consumes
// 0% CPU and schedules no GPU kernels. TTS is a subprocess that exists only
// while speaking.
package pipeline

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/fsm"
	"github.com/ibnHatab/fsttm/go/internal/intent"
	"github.com/ibnHatab/fsttm/go/internal/llm"
	"github.com/ibnHatab/fsttm/go/internal/stt"
	"github.com/ibnHatab/fsttm/go/internal/tts"
	"github.com/ibnHatab/fsttm/go/internal/vad"
)

// bargeGrace suppresses barge-in for the first moments of playback (TTS
// onset transient / leftover VAD frames from the user's own utterance).
const bargeGrace = 600 * time.Millisecond

type Config struct {
	SystemPrompt string
	GBNF         string
	WakeWord     string // "" → always awake
	// BargeIn allows a VAD onset to cut TTS. Enable ONLY with echo
	// cancellation in the audio path (AEC virtual mic / USB conference
	// speakerphone); without it the mic hears the TTS and self-triggers.
	// Off = half-duplex: mic events are ignored while the system speaks.
	BargeIn bool
}

type Engine struct {
	cfg  Config
	LLM  *llm.LLM
	STT  *stt.STT
	TTS  *tts.TTS
	Dsp  *intent.Dispatcher
	Turn *fsm.Model

	awake bool

	speakReq  chan string
	ttsDone   chan bool
	speakingT time.Time
	speakEndT time.Time
	speaking  bool
}

func New(cfg Config, l *llm.LLM, s *stt.STT, t *tts.TTS) *Engine {
	return &Engine{
		cfg: cfg, LLM: l, STT: s, TTS: t,
		Dsp:      intent.NewLogging(),
		Turn:     fsm.New(),
		awake:    cfg.WakeWord == "",
		speakReq: make(chan string),
		ttsDone:  make(chan bool),
	}
}

// Run is the orchestrator loop. vadEvents may be nil (headless); textIn may
// be nil (voice-only). Blocks until ctx is done.
func (e *Engine) Run(ctx context.Context, vadEvents <-chan vad.Event, textIn <-chan string) {
	// speaker goroutine: exists to keep the orchestrator responsive (barge-in
	// events keep flowing while a subprocess plays audio).
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case text := <-e.speakReq:
				ok, err := e.TTS.Speak(ctx, text)
				if err != nil {
					log.Printf("[tts] %v", err)
				}
				select {
				case e.ttsDone <- ok:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return

		case ev, open := <-vadEvents:
			if !open {
				vadEvents = nil
				if textIn == nil {
					return // all sources gone
				}
				continue
			}
			e.onVadEvent(ev)

		case ok := <-e.ttsDone:
			e.speaking = false
			e.speakEndT = time.Now()
			_ = ok
			// Response finished (or was cut): release the floor.
			_ = e.Turn.SystemAction('R')

		case text, open := <-textIn:
			if !open {
				textIn = nil
				if vadEvents == nil {
					// headless EOF: let the last narration finish, then exit
					if e.speaking {
						<-e.ttsDone
					}
					return
				}
				continue
			}
			// Headless is turn-disciplined: finish the running narration
			// before the next simulated user turn (the mic path gets this
			// for free from half-duplex).
			for e.speaking {
				select {
				case <-e.ttsDone:
					e.speaking = false
					_ = e.Turn.SystemAction('R')
				case <-ctx.Done():
					return
				}
			}
			// Simulate the user turn (grab + release).
			_ = e.Turn.UserAction('G')
			_ = e.Turn.UserAction('R')
			e.handleUtterance(text)
		}
	}
}

func (e *Engine) onVadEvent(ev vad.Event) {
	if ev.SpeechStart {
		if e.speaking {
			if e.cfg.BargeIn && time.Since(e.speakingT) > bargeGrace {
				log.Print("[barge-in] cutting TTS")
				e.TTS.Cancel()
				_ = e.Turn.UserAction('G')   // SYSTEM → BOTHs
				_ = e.Turn.SystemAction('R') // BOTHs → USER
			}
			return // half-duplex: mic is the system's own voice
		}
		_ = e.Turn.UserAction('G')
		return
	}
	if ev.Utterance == nil {
		return
	}
	if e.speaking && !e.cfg.BargeIn {
		return // utterance captured during playback = self-echo; drop
	}
	// Echo tail: an utterance whose AUDIO STARTED while the system was still
	// speaking is the system's own voice — the VAD only closes it ~padding ms
	// after the reply ends, when `speaking` is already false. Reconstruct the
	// onset from the utterance length and drop it if it overlaps playback.
	uttDur := time.Duration(len(ev.Utterance)/2) * time.Second / vad.SampleRate
	if !e.cfg.BargeIn && time.Since(e.speakEndT)-uttDur < 200*time.Millisecond {
		log.Printf("[echo] dropped %.1fs utterance overlapping own speech", uttDur.Seconds())
		return
	}
	_ = e.Turn.UserAction('R')
	if !e.Turn.IsUser() {
		return // floor gate
	}
	res, ok := e.STT.Transcribe(ev.Utterance)
	if !ok || res.Parasite {
		return
	}
	log.Printf("[user] %q  [FSM:%s]", res.Text, e.Turn.State)
	e.handleUtterance(res.Text)
}

func (e *Engine) handleUtterance(text string) {
	// attention layer (lite): asleep until the wake word; stays awake.
	if !e.awake {
		norm := strings.ToLower(text)
		if !strings.Contains(norm, strings.ToLower(e.cfg.WakeWord)) {
			log.Printf("[asleep] ignored (say %q first): %q", e.cfg.WakeWord, text)
			return
		}
		e.awake = true
		log.Printf("[attention] awake")
		// strip everything up to and including the wake word
		i := strings.Index(norm, strings.ToLower(e.cfg.WakeWord))
		text = strings.TrimLeft(text[i+len(e.cfg.WakeWord):], " ,.!?")
		if text == "" {
			e.say("Yes?")
			return
		}
	}

	if !e.Turn.IsUser() {
		return
	}
	if err := e.Turn.SystemAction('G'); err != nil {
		return
	}

	r, err := e.LLM.TwoPass(e.cfg.SystemPrompt, text, e.cfg.GBNF)
	if err != nil {
		log.Printf("[llm] %v", err)
		_ = e.Turn.SystemAction('R')
		return
	}
	log.Printf("[intent] %s (json=%dms tts=%dms)", r.JSON,
		r.TJSON.Milliseconds(), r.TTTS.Milliseconds())

	cmd, err := intent.Parse(r.JSON)
	if err != nil {
		log.Printf("[intent] %v", err)
		e.say(r.Voice)
		return
	}

	spoken := ""
	switch cmd.Meta() {
	case "TIME":
		spoken = "It's " + time.Now().Format("3:04 PM") + "."
	case "DATE":
		spoken = "It's " + time.Now().Format("Monday, January 2") + "."
	case "CHITCHAT", "UNKNOWN":
		spoken = r.Voice
	default:
		spoken = e.Dsp.Handle(cmd) // deterministic QUERY answer, or ""
		if spoken == "" {
			spoken = r.Voice
		}
	}
	e.say(spoken)
}

// say hands the narration to the speaker goroutine; the FSM floor is
// released when ttsDone comes back.
func (e *Engine) say(text string) {
	text = sanitizeAck(text)
	if strings.TrimSpace(text) == "" {
		_ = e.Turn.SystemAction('R')
		return
	}
	fmt.Printf("  voice → %q\n", text)
	e.speaking = true
	e.speakingT = time.Now()
	e.speakReq <- text
}

// sanitizeAck keeps the spoken ack speakable: the pass-2 model occasionally
// parrots the intent JSON instead of a natural sentence — never speak that
// (the mic would hear it and re-command). Placeholders likewise.
func sanitizeAck(text string) string {
	t := strings.TrimSpace(strings.Trim(strings.TrimSpace(text), `"`))
	if t == "" {
		return ""
	}
	if strings.ContainsAny(t, "{}[]<>") || len(t) > 200 {
		return "Okay."
	}
	return t
}
