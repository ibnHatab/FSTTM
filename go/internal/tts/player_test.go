package tts

import (
	"context"
	"testing"
	"time"
)

func silence(d time.Duration) []byte {
	n := int(d.Seconds() * 24000)
	return WAV16(make([]float32, n), 24000)
}

// A Cancel during the synthesis (before the player starts) cuts the utterance; one during
// the playback kills the player; an uncancelled one completes.
func TestPlayerCancel(t *testing.T) {
	p := wavPlayer{Player: []string{"sh", "-c", "sleep 0.3", "player"}} // a stand-in player: "plays" 0.3 s, ignores the file
	ctx := context.Background()
	gen := p.begin()
	p.Cancel()
	if r, _ := p.play(ctx, silence(time.Second), gen); r.Completed || r.Played != 0 || r.Synthesized != time.Second {
		t.Errorf("cut while synthesized: %+v", r)
	}
	gen = p.begin()
	go func() { time.Sleep(100 * time.Millisecond); p.Cancel() }()
	if r, _ := p.play(ctx, silence(time.Second), gen); r.Completed || r.Played < 50*time.Millisecond || r.Played > 250*time.Millisecond {
		t.Errorf("cut while played: %+v", r)
	}
	if r, err := p.play(ctx, silence(time.Second), p.begin()); !r.Completed || err != nil {
		t.Errorf("played: %+v %v", r, err)
	}
}
