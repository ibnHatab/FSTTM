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
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/attention"
	"github.com/ibnHatab/fsttm/go/internal/fsm"
	"github.com/ibnHatab/fsttm/go/internal/intent"
	"github.com/ibnHatab/fsttm/go/internal/llm"
	"github.com/ibnHatab/fsttm/go/internal/stt"
	"github.com/ibnHatab/fsttm/go/internal/tts"
	"github.com/ibnHatab/fsttm/go/internal/vad"
)

// Driver seams — small interfaces so e2e tests can fake every stage while
// the production wiring passes the real drivers unchanged.
type Transcriber interface {
	Transcribe(pcm []byte) (stt.Result, bool)
}

type IntentGen interface {
	TwoPass(systemPrompt, userText, gbnf string) (*llm.Result, error)
}

// ChatTurn is a free conversation turn (the cloud agent): sentences as the reply streams
// in, the full reply at the end. A cancelled ctx ends it (a barge-in).
type ChatTurn interface {
	Turn(ctx context.Context, text string, onSentence func(string)) (string, error)
}

// OwnerVerifier gates utterances on the imprinted owner's voice
// (voiceid.SpeakerVerifier satisfies this; nil → open to everyone).
type OwnerVerifier interface {
	IsOwner(pcm []byte) (bool, float64)
}

// bargeGrace suppresses barge-in for the first moments of playback (TTS
// onset transient / leftover VAD frames from the user's own utterance).
const bargeGrace = 600 * time.Millisecond

type Config struct {
	SystemPrompt string
	GBNF         string

	// Attention layer (wake word / sleep) — see internal/attention.
	Attention    bool     // false → always awake
	WakeWords    []string // "hello nina", "hey nina", "nina", …
	SleepPhrases []string // "go to sleep", "voice off", … (wake-prefixed only)

	// BargeIn mode — how a user may cut running narration:
	//   "off"     half-duplex (default): mic events ignored while speaking,
	//             overlapping utterances dropped as echo. No AEC needed.
	//   "vad"     a bare VAD onset after the grace window cuts TTS. Needs
	//             clean echo cancellation (AEC virtual mic / hardware AEC),
	//             else the robot's own voice self-triggers.
	//   "confirm" soft-duck (the Python engine's design): narration keeps
	//             playing; the overlapping utterance is transcribed and —
	//             when an owner is imprinted — voice-verified. Only a real,
	//             owner-confirmed transcript cuts TTS. Works with imperfect
	//             AEC: residue rarely survives STT noise filters, and the
	//             robot's own TTS voice can never match the owner imprint.
	BargeIn string

	// OwnerGate — which utterances require the imprinted owner's voice:
	//   "off" | "wake" (imprinting: only the owner can wake it) | "always".
	// Barge-in confirmation ALWAYS verifies when a verifier is present.
	OwnerGate string
}

type Engine struct {
	cfg   Config
	LLM   IntentGen
	STT   Transcriber
	TTS   tts.Speaker
	Dsp   *intent.Dispatcher
	Turn  *fsm.Model
	Attn  *attention.Attention
	Owner OwnerVerifier // nil → open to everyone
	// Events, when set, receives engine happenings for external consumers
	// (the nina relay → /nina/intent, /nina/dialog_state):
	//   kind "intent":  payload = *llm.Result (after a completed turn)
	//   kind "dialog":  payload = string state (ASLEEP/AWAKE/LISTENING/SPEAKING)
	Events func(kind string, payload any)
	// Chat, when set, answers every turn as free conversation (the cloud agent); the
	// intent LLM is then unused (may be nil).
	Chat ChatTurn
	// Typed carries text turns from a page (the panel's text line): a person's turn,
	// addressed to her (no wake word needed).
	Typed chan string

	speakReq   chan string
	streamReq  chan (<-chan string)
	turnCancel context.CancelFunc
	uttEnd     time.Time   // when the last utterance closed (the end-to-end clock)
	cut        atomic.Bool // the running streamed reply was cut: speak no more of it
	ttsDone    chan bool
	speakingT  time.Time
	speakEndT  time.Time
	speaking   bool

	announceCh chan string // system-initiated narration (warnings, status)
	pending    []string    // announcements deferred while the floor is busy

	// the page's switches (Control): the mic off drops what the VAD brings (nothing is
	// transcribed); the voice off speaks nothing (the replies are text only)
	micOff   atomic.Bool
	voiceOff atomic.Bool
	ctl      chan string
}

// Control is a switch from a page, from any goroutine: "mic on" | "mic off" | "voice on" |
// "voice off" | "stop" (hush the narration now: the reply's text still comes). The voice off
// also hushes what is being said.
func (e *Engine) Control(op string) {
	switch op {
	case "mic on", "mic off":
		e.micOff.Store(op == "mic off")
		e.event("switch", map[string]any{"mic": op == "mic on", "voice": !e.voiceOff.Load()})
		return
	case "voice on", "voice off":
		e.voiceOff.Store(op == "voice off")
		e.event("switch", map[string]any{"mic": !e.micOff.Load(), "voice": op == "voice on"})
		if op == "voice on" {
			return
		}
	case "stop":
	default:
		return
	}
	select {
	case e.ctl <- "hush":
	default:
	}
}

// Mic and Voice are the switches' states.
func (e *Engine) Mic() bool   { return !e.micOff.Load() }
func (e *Engine) Voice() bool { return !e.voiceOff.Load() }

// hush stops the narration at once (the barge-in cut: the sentence playing, the rest of the
// streamed reply unspoken); the LLM's reply is not cancelled, its text still comes.
func (e *Engine) hush() {
	if !e.speaking {
		return
	}
	log.Print("[tts] hushed from the page")
	e.cut.Store(true)
	e.TTS.Cancel()
}

func New(cfg Config, l IntentGen, s Transcriber, t tts.Speaker) *Engine {
	if cfg.BargeIn == "" {
		cfg.BargeIn = "off"
	}
	return &Engine{
		cfg: cfg, LLM: l, STT: s, TTS: t,
		Dsp:  intent.NewLogging(),
		Turn: fsm.New(),
		Attn: attention.New(cfg.Attention, cfg.WakeWords,
			cfg.SleepPhrases, true),
		speakReq:   make(chan string),
		streamReq:  make(chan (<-chan string)),
		ttsDone:    make(chan bool),
		announceCh: make(chan string, 8),
		ctl:        make(chan string, 4),
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
			case sentences := <-e.streamReq:
				// a streamed reply: each sentence spoken as it arrives; a cut ends the rest
				done, first := true, true
				for s := range sentences {
					if !done || e.cut.Load() {
						done = false
						continue // drain after a cut
					}
					if e.voiceOff.Load() {
						continue // the voice is off: the reply is text only
					}
					if first && !e.uttEnd.IsZero() {
						log.Printf("[timing] first audio %.2fs after the end of speech", time.Since(e.uttEnd).Seconds())
						e.event("speak", map[string]any{"first_audio_s": time.Since(e.uttEnd).Seconds(), "text": s})
						first = false
					}
					res, err := e.TTS.Speak(ctx, s)
					if err != nil {
						log.Printf("[tts] %v", err)
					}
					if !res.Completed && err == nil {
						done = false
						log.Printf("[tts] cut at %.0f%% of %q", res.Fraction()*100, s)
						e.event("cut", map[string]any{"fraction": res.Fraction(), "text": s})
					}
				}
				select {
				case e.ttsDone <- done:
				case <-ctx.Done():
					return
				}
			case text := <-e.speakReq:
				if e.voiceOff.Load() {
					select {
					case e.ttsDone <- true:
					case <-ctx.Done():
						return
					}
					continue
				}
				res, err := e.TTS.Speak(ctx, text)
				if err != nil {
					log.Printf("[tts] %v", err)
				}
				if !res.Completed && res.Synthesized > 0 {
					// exact fraction heard — the replay/skip input
					log.Printf("[tts] cut at %.0f%% (%.2fs of %.2fs)",
						res.Fraction()*100, res.Played.Seconds(),
						res.Synthesized.Seconds())
				}
				select {
				case e.ttsDone <- res.Completed:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	for {
		e.tryAnnounce() // deferred announcements go out once the floor frees

		select {
		case <-ctx.Done():
			return

		case <-e.ctl:
			e.hush()

		case text := <-e.announceCh:
			e.pending = append(e.pending, text)
			// dispatched by tryAnnounce at the top of the loop

		case ev, open := <-vadEvents:
			if !open {
				vadEvents = nil
				if textIn == nil {
					return // all sources gone
				}
				continue
			}
			e.onVadEvent(ev)

		case text := <-e.Typed:
			if e.speaking { // a typed line cuts the running reply
				e.cutReply()
			}
			_ = e.Turn.UserAction('G')
			_ = e.Turn.UserAction('R')
			e.event("typed", text)
			e.uttEnd = time.Now()
			e.Attn.Touch()
			e.handleUtterance(text)

		case ok := <-e.ttsDone:
			e.speaking = false
			e.speakEndT = time.Now()
			e.Attn.Touch() // her reply is activity: the idle time starts after it
			_ = ok
			// Response finished (or was cut): release the floor.
			_ = e.Turn.SystemAction('R')
			e.event("dialog", "LISTENING")

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
			e.processUtterance(text, nil)
		}
	}
}

func (e *Engine) onVadEvent(ev vad.Event) {
	if e.micOff.Load() {
		return // the mic is off: nothing heard (the capture keeps running, its frames are silence)
	}
	if ev.SpeechStart {
		if e.speaking {
			if e.cfg.BargeIn == "vad" && time.Since(e.speakingT) > bargeGrace {
				log.Print("[barge-in] VAD onset — cutting TTS")
				e.cancelTurnCtx()
				e.TTS.Cancel()
				_ = e.Turn.UserAction('G')   // SYSTEM → BOTHs (transition 7)
				_ = e.Turn.SystemAction('R') // BOTHs → USER  (transition 8)
			}
			// "off": half-duplex — mic is the system's own voice.
			// "confirm": tentative — narration keeps playing until the
			// utterance closes and a real transcript confirms.
			return
		}
		_ = e.Turn.UserAction('G')
		return
	}
	if ev.Utterance == nil {
		return
	}
	uttDur := time.Duration(len(ev.Utterance)/2) * time.Second / vad.SampleRate

	if e.speaking {
		if e.cfg.BargeIn == "confirm" {
			e.confirmBargeIn(ev.Utterance)
		}
		return // "off"/"vad": utterance during playback is self-echo
	}
	// Echo tail (half-duplex only): an utterance whose AUDIO STARTED while
	// the system was still speaking is the system's own voice — the VAD only
	// closes it ~padding ms after the reply ends, when `speaking` is already
	// false. Reconstruct the onset and drop it if it overlaps playback.
	if e.cfg.BargeIn == "off" &&
		time.Since(e.speakEndT)-uttDur < 200*time.Millisecond {
		log.Printf("[echo] dropped %.1fs utterance overlapping own speech",
			uttDur.Seconds())
		e.event("echo", map[string]any{"audio_s": uttDur.Seconds()})
		return
	}
	_ = e.Turn.UserAction('R')
	if !e.Turn.IsUser() {
		return // floor gate
	}
	t0 := time.Now()
	res, ok := e.STT.Transcribe(ev.Utterance)
	log.Printf("[stt] %.2fs for %.1fs of audio%s%s", time.Since(t0).Seconds(), uttDur.Seconds(),
		map[bool]string{true: " (cut at the maximum length)"}[ev.Forced],
		map[bool]string{true: " (looped: decoded again)"}[res.Redecoded])
	e.event("stt", map[string]any{"stt_s": time.Since(t0).Seconds(), "audio_s": uttDur.Seconds(), "text": res.Text,
		"kept": ok && !res.Parasite && !res.Doubtful, "forced": ev.Forced, "redecoded": res.Redecoded, "doubtful": res.Doubtful})
	if !ok || res.Parasite {
		return
	}
	if res.Doubtful {
		// the decoder looped even decoding carefully: not a turn (a person never says a phrase
		// twenty times); shown, not acted on
		log.Printf("[stt] doubtful (a loop), not a turn: %q", Short(res.Text))
		e.event("doubtful", Short(res.Text))
		return
	}
	e.uttEnd = t0
	log.Printf("[user] %q  [FSM:%s]", res.Text, e.Turn.State)
	e.event("heard", res.Text)
	e.processUtterance(res.Text, ev.Utterance)
}

// confirmBargeIn is the soft-duck design ported from the Python engine: the
// VAD blip alone is only TENTATIVE (narration keeps playing); a real
// transcript — noise-filtered AND owner-verified when an imprint exists —
// CONFIRMS the barge-in. Only then is the output of the unfinished
// utterance cut (librhvoice mid-stream abort, exact fraction logged) and
// the floor walked SYSTEM →(K,G) BOTHs →(R,K) USER (transitions 7+8). The
// robot's own TTS voice cannot confirm: whisper's noise filters eat AEC
// residue, and the voice imprint rejects what survives.
func (e *Engine) confirmBargeIn(pcm []byte) {
	res, ok := e.STT.Transcribe(pcm)
	if !ok || res.Parasite {
		log.Print("[barge-in] tentative blip was noise — narration continues")
		return
	}
	if e.Owner != nil {
		if ok, score := e.Owner.IsOwner(pcm); !ok {
			log.Printf("[barge-in] voice not the owner (%.2f) — "+
				"narration continues", score)
			return
		}
	}
	log.Printf("[barge-in] CONFIRMED by %q — cutting TTS", res.Text)
	e.cancelTurnCtx()
	e.TTS.Cancel()
	// drain the speaker's completion so floor accounting stays single-owner
	select {
	case <-e.ttsDone:
	case <-time.After(2 * time.Second):
	}
	e.speaking = false
	e.speakEndT = time.Now()
	_ = e.Turn.UserAction('G')   // SYSTEM → BOTHs (transition 7)
	_ = e.Turn.SystemAction('R') // BOTHs → USER  (transition 8)
	e.processUtterance(res.Text, pcm)
}

// processUtterance runs the attention + owner gates, then the intent turn.
// pcm may be nil (headless) — voice verification is then skipped.
func (e *Engine) processUtterance(text string, pcm []byte) {
	if !e.Attn.Awake() {
		// ASLEEP: only the wake word acts — and when an owner is imprinted,
		// only in the OWNER's voice (the robot imprints; ownership transfer
		// = replace the profile file).
		if matched, _ := e.Attn.MatchWake(text); !matched {
			log.Printf("[asleep] ignored: %q", Short(text))
			e.event("asleep", Short(text))
			return
		}
		if e.ownerGated("wake") && pcm != nil && e.Owner != nil {
			if ok, score := e.Owner.IsOwner(pcm); !ok {
				log.Printf("[asleep] wake word in a stranger's voice "+
					"(%.2f) — staying asleep", score)
				return
			}
		}
		d := e.Attn.OnUtterance(text) // wakes
		log.Print("[attention] awake")
		e.event("dialog", "AWAKE")
		if d.Text == "" {
			e.grabAndSay("Yes?")
			return
		}
		text = d.Text
	} else {
		if e.ownerGated("always") && pcm != nil && e.Owner != nil {
			if ok, score := e.Owner.IsOwner(pcm); !ok {
				log.Printf("[owner] utterance rejected (%.2f): %q", score, text)
				return
			}
		}
		d := e.Attn.OnUtterance(text)
		switch d.Action {
		case "sleep":
			log.Print("[attention] → ASLEEP")
			e.event("dialog", "ASLEEP")
			e.grabAndSay("Voice controls disabled.")
			return
		case "ignore":
			return
		}
		text = d.Text
	}
	e.handleUtterance(text)
}

// ownerGated reports whether `stage` requires the owner's voice.
func (e *Engine) ownerGated(stage string) bool {
	switch e.cfg.OwnerGate {
	case "always":
		return true
	case "wake":
		return stage == "wake"
	}
	return false
}

// grabAndSay speaks a short system phrase with proper floor handling.
func (e *Engine) grabAndSay(text string) {
	if err := e.Turn.SystemAction('G'); err != nil {
		return
	}
	e.say(text)
}

func (e *Engine) handleUtterance(text string) {
	if !e.Turn.IsUser() {
		return
	}
	if err := e.Turn.SystemAction('G'); err != nil {
		return
	}
	if e.Chat != nil {
		e.chatTurn(text)
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
	if e.Events != nil {
		e.Events("intent", r)
	}

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

// chatTurn is a free conversation turn: the reply streams in sentence by sentence and each
// is spoken as it arrives (the speaker goroutine); the floor is released when the last one
// has played (ttsDone). The orchestrator stays free meanwhile (barge-in, typed lines).
func (e *Engine) chatTurn(text string) {
	ctx, cancel := context.WithCancel(context.Background())
	e.turnCancel = cancel
	e.cut.Store(false)
	sentences := make(chan string, 32)
	start := time.Now()
	go func() {
		defer close(sentences)
		firstAt := time.Duration(0)
		full, err := e.Chat.Turn(ctx, text, func(s string) {
			if firstAt == 0 {
				firstAt = time.Since(start)
			}
			select {
			case sentences <- s:
			case <-ctx.Done():
			}
		})
		if err != nil && ctx.Err() == nil {
			log.Printf("[chat] %v", err)
			if full == "" {
				full = "Sorry, I can't think right now."
				sentences <- full
			}
		}
		log.Printf("[timing] llm first sentence %.2fs, whole reply %.2fs: %q",
			firstAt.Seconds(), time.Since(start).Seconds(), full)
		e.event("llm", map[string]any{"first_sentence_s": firstAt.Seconds(), "whole_s": time.Since(start).Seconds()})
		e.event("reply", full)
	}()
	e.speaking = true
	e.speakingT = time.Now()
	e.event("dialog", "SPEAKING")
	e.streamReq <- sentences
}

func (e *Engine) cancelTurnCtx() {
	e.cut.Store(true)
	if e.turnCancel != nil {
		e.turnCancel()
		e.turnCancel = nil
	}
}

// cutReply stops the running reply (the rest of the stream, the sentence playing) and
// waits for the speaker to give the floor back.
func (e *Engine) cutReply() {
	e.cancelTurnCtx()
	e.TTS.Cancel()
	select {
	case <-e.ttsDone:
	case <-time.After(3 * time.Second):
	}
	e.speaking = false
	e.speakEndT = time.Now()
	_ = e.Turn.UserAction('G')
	_ = e.Turn.SystemAction('R')
}

// say hands the narration to the speaker goroutine; the FSM floor is
// released when ttsDone comes back.
func (e *Engine) say(text string) {
	text = sanitizeAck(text)
	if strings.TrimSpace(text) == "" {
		_ = e.Turn.SystemAction('R')
		return
	}
	fmt.Fprintf(os.Stderr, "  voice → %q\n", text)
	e.speaking = true
	e.speakingT = time.Now()
	e.event("dialog", "SPEAKING")
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

// Announce queues SYSTEM-INITIATED narration — boot greetings, battery
// warnings, robot status. Per the paper this is transition 5
// (FREE —(G,W)→ SYSTEM, cost 0 in Table 1): the system may freely take a
// floor nobody claims, but it never cuts the user — while the user holds
// the floor the announcement waits. Safe to call from any goroutine.
func (e *Engine) Announce(text string) {
	select {
	case e.announceCh <- text:
	default: // queue full — drop rather than block a robot control path
		log.Printf("[announce] queue full, dropped %q", text)
	}
}

// tryAnnounce dispatches the oldest pending announcement when the floor is
// free (FREEu/FREEs) and nothing is playing.
func (e *Engine) tryAnnounce() {
	if len(e.pending) == 0 || e.speaking {
		return
	}
	if e.Turn.State != fsm.FreeU && e.Turn.State != fsm.FreeS {
		return // user holds (or contests) the floor — never cut them
	}
	if err := e.Turn.SystemAction('G'); err != nil {
		return
	}
	text := e.pending[0]
	e.pending = e.pending[1:]
	log.Printf("[announce] %q", text)
	e.say(text)
}

func (e *Engine) event(kind string, payload any) {
	if e.Events != nil {
		e.Events(kind, payload)
	}
}

// Short is a transcript for a note: its first 160 characters, at a word, and an ellipsis.
func Short(text string) string {
	const n = 160
	r := []rune(text)
	if len(r) <= n {
		return text
	}
	cut := n
	for i := n; i > n-30; i-- {
		if r[i] == ' ' {
			cut = i
			break
		}
	}
	return strings.TrimRight(string(r[:cut]), " ,.;") + " …"
}
