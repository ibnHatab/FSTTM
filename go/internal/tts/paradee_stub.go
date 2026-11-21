//go:build noparadee

package tts

import (
	"context"
	"errors"
)

// Paradee is not built in (-tags noparadee: no onnxruntime / espeak-ng on this host).
type Paradee struct{ wavPlayer }

func NewParadee(dir, voice string, speed float64, threads int, sink string) (*Paradee, error) {
	return nil, errors.New("paradee: built with -tags noparadee")
}

func (p *Paradee) Speak(ctx context.Context, text string) (SpeakResult, error) {
	return SpeakResult{}, errors.New("paradee: built with -tags noparadee")
}
