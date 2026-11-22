// Package stt wraps the OFFICIAL whisper.cpp Go binding (bindings/go —
// in-tree with the whisper.cpp checkout, so always API-current; verified on
// par with pywhispercpp: 42 ms vs 44 ms for 11 s of audio on the same
// CUDA libwhisper).
//
// Transcriptions run only when an utterance arrives — the model sits idle
// (0 CPU, 0 GPU kernels) between utterances.
package stt

import (
	"encoding/binary"
	"log"
	"regexp"
	"strings"
	"sync"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// Annotation-only transcripts ("(sighs)", "[BLANK_AUDIO]", "*cough*") are
// noise, never speech — same filter as fsttm/whisper.py.
var annotationRe = regexp.MustCompile(`[\[(][^\])]*[\])]|\*[^*]*\*`)

type STT struct {
	// AudioCtx shortens the encoder's window for short utterances (0: whisper's 1500;
	// 768 on the NUC: base.en 2.34 s -> 1.21 s for 2.5 s of speech, the same text). It is a
	// floor: an utterance longer than the window it covers (50 frames a second) gets the window
	// it needs (CtxFor), up to the full 1500, so the encoder never sees a truncated clip.
	AudioCtx uint
	Threads  uint
	// NoFallback: no temperature fallback (a decode that fails its thresholds is kept, not
	// re-run hotter): bounds the time of a short utterance.
	NoFallback bool
	mu    sync.Mutex // whisper contexts are not goroutine-safe
	model whisper.Model
	lang  string
	// Parasite phrases — whole-string hallucinations on silence.
	parasites map[string]bool
}

func New(modelPath, lang string, parasites []string) (*STT, error) {
	m, err := whisper.New(modelPath)
	if err != nil {
		return nil, err
	}
	p := map[string]bool{"thank you": true, "thanks": true}
	for _, s := range parasites {
		p[strings.ToLower(strings.TrimSpace(s))] = true
	}
	s := &STT{model: m, lang: lang, parasites: p}
	// Warm the CUDA kernels so the first real utterance is fast.
	_, _ = s.transcribe(make([]float32, 16000), false)
	return s, nil
}

func (s *STT) Close() { s.model.Close() }

// Result of one utterance transcription.
type Result struct {
	Text     string
	Parasite bool // likely hallucination; only meaningful for barge-in confirm
	// Doubtful: the decoder looped (a phrase repeated over and over) even after a second,
	// careful decode; Text has the runs collapsed. Not a turn to act on.
	Doubtful bool
	// Redecoded: the first decode looped and a careful one (the full window, the temperature
	// fallback, no carried context) was run.
	Redecoded bool
}

// Transcribe converts one utterance (16 kHz mono s16le PCM) to text.
// Returns ok=false for noise (annotations, too short).
func (s *STT) Transcribe(pcm []byte) (Result, bool) {
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768.0
	}
	text, err := s.transcribe(samples, false)
	if err != nil {
		return Result{}, false
	}
	text = strings.TrimSpace(text)
	var r Result
	if Looped(text) {
		// whisper's guard against loops is the temperature fallback, off on the NUC for speed:
		// decode this one again carefully
		r.Redecoded = true
		again, err := s.transcribe(samples, true)
		if err == nil {
			again = strings.TrimSpace(again)
			log.Printf("[stt] loop in the first decode (%d chars); careful decode: %d chars, looped=%v",
				len(text), len(again), Looped(again))
			text = again
		}
		if Looped(text) {
			r.Doubtful = true
			text = Collapse(text)
		}
	}
	if isHardNoise(text) {
		return Result{}, false
	}
	norm := strings.ToLower(strings.TrimRight(text, ".!?, "))
	r.Text, r.Parasite = text, s.parasites[norm]
	return r, true
}

// CtxFor is the encoder window for n samples: the configured floor, or what the clip needs
// (50 frames a second, 64 to spare, rounded up to 64), whole (0: 1500) when that reaches 1500.
func CtxFor(floor uint, n int) uint {
	if floor == 0 {
		return 0
	}
	need := uint((n*50+15999)/16000) + 64
	need = (need + 63) / 64 * 64
	if need < floor {
		need = floor
	}
	if need >= 1500 {
		return 0
	}
	return need
}

func (s *STT) transcribe(samples []float32, careful bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctx, err := s.model.NewContext()
	if err != nil {
		return "", err
	}
	_ = ctx.SetLanguage(s.lang)
	th := s.Threads
	if th == 0 {
		th = 6
	}
	ctx.SetThreads(th)
	if careful {
		// the full window, the temperature fallback (whisper's own loop guard: a decode whose
		// compression ratio or log-probability fails is re-run hotter), no text carried from
		// one 30 s window into the next (the usual seed of a loop)
		ctx.SetMaxContext(0)
	} else {
		if c := CtxFor(s.AudioCtx, len(samples)); c > 0 {
			ctx.SetAudioCtx(c)
		}
		if s.NoFallback {
			ctx.SetTemperatureFallback(-1)
		}
	}
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return "", err
	}
	var b strings.Builder
	for {
		seg, err := ctx.NextSegment()
		if err != nil {
			break
		}
		b.WriteString(seg.Text)
	}
	return b.String(), nil
}

func isHardNoise(text string) bool {
	t := strings.TrimSpace(text)
	if len(t) < 2 {
		return true
	}
	return len(strings.TrimSpace(annotationRe.ReplaceAllString(t, ""))) < 2
}
