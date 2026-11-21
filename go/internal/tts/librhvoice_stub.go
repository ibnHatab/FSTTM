//go:build norhvoice

package tts

import (
	"context"
	"errors"
)

// LibEngine without librhvoice (build tag norhvoice: the NUC speaks with Piper, engine
// piper); NewLib fails, so NewSpeaker falls back to the RHVoice-client subprocess.
type LibEngine struct{}

func NewLib(Config) (*LibEngine, error) {
	return nil, errors.New("tts: built with -tags norhvoice (no librhvoice)")
}

func (e *LibEngine) Speak(context.Context, string) (SpeakResult, error) {
	return SpeakResult{}, errors.New("tts: no librhvoice")
}

func (e *LibEngine) Cancel() {}
