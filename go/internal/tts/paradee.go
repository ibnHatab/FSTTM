//go:build !noparadee

package tts

import (
	"context"
	"strings"

	"github.com/ibnHatab/fsttm/go/internal/paradee"
)

// Paradee speaks with the Paradee model in-process (internal/paradee: misaki's G2P in Go,
// onnxruntime), the voice of the user's narration script (af_heart), and plays each
// utterance's WAV like Piper (paplay to the default sink; Cancel kills the player). The
// engine already hands it one sentence at a time.
type Paradee struct {
	M     *paradee.Model
	Voice string
	Speed float32
	wavPlayer
}

// NewParadee loads the model and lexicons from dir (models/paradee).
func NewParadee(dir, voice string, speed float64, threads int, sink string) (*Paradee, error) {
	m, err := paradee.Open(dir, threads)
	if err != nil {
		return nil, err
	}
	if voice == "" {
		voice = "af_heart"
	}
	if speed <= 0 {
		speed = 1
	}
	return &Paradee{M: m, Voice: voice, Speed: float32(speed), wavPlayer: newWavPlayer(sink)}, nil
}

func (p *Paradee) Speak(ctx context.Context, text string) (SpeakResult, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return SpeakResult{Completed: true}, nil
	}
	gen := p.begin()
	audio, err := p.M.Synthesize(text, p.Speed, p.Voice)
	if err != nil {
		return SpeakResult{}, err
	}
	return p.play(ctx, WAV16(audio, paradee.SampleRate), gen)
}
