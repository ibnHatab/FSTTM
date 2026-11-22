package vad

import (
	"context"
	"os"
	"testing"
)

// Continuous speech longer than MaxMs is cut into pieces no longer than MaxMs, each a Forced
// utterance followed at once by a new SpeechStart; the last piece closes on the silence after.
func TestMaxUtterance(t *testing.T) {
	b, err := os.ReadFile("../../../samples/jfk.wav") // 11 s of speech, 16 kHz mono s16
	if err != nil {
		t.Skip(err)
	}
	pcm := b[44:]
	s, err := New(2, 300)
	if err != nil {
		t.Fatal(err)
	}
	s.MaxMs = 2000
	frames := make(chan []byte, 4096)
	for i := 0; i+FrameBytes <= len(pcm); i += FrameBytes {
		frames <- pcm[i : i+FrameBytes]
	}
	for i := 0; i < 100; i++ { // 2 s of silence closes the last utterance
		frames <- make([]byte, FrameBytes)
	}
	close(frames)
	events := make(chan Event, 64)
	s.Run(context.Background(), frames, events)
	forced, utts, starts := 0, 0, 0
	for ev := range events {
		if ev.SpeechStart {
			starts++
		}
		if ev.Utterance != nil {
			utts++
			if ev.Forced {
				forced++
			}
			if ms := len(ev.Utterance) / FrameBytes * FrameMs; ms > s.MaxMs+FrameMs*20 {
				t.Errorf("an utterance of %d ms, max %d", ms, s.MaxMs)
			}
		}
	}
	if forced < 2 || utts < forced || starts < utts {
		t.Fatalf("forced %d, utterances %d, starts %d", forced, utts, starts)
	}
	t.Logf("forced %d of %d utterances", forced, utts)
}
