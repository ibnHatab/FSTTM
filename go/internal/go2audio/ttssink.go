package go2audio

// Go2Speaker adapts the WebRTC streaming path to the engine's tts.Speaker
// interface: synthesize with RHVoice, stream the PCM to the robot speaker,
// report the exact fraction heard (barge-in needs it — same contract as the
// ALSA/librhvoice engines). Selected by config tts.engine: "go2".
//
// Kept in this package (not internal/tts) so pion/libopus stay out of the
// default build; only deployments that dial the robot pull them in.

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/tts"
)

var _ tts.Speaker = (*Go2Speaker)(nil)

type Go2Speaker struct {
	player *Player
	client string // RHVoice-client
	voice  string
	rate   float64
	volume float64
	mu     sync.Mutex
}

// NewGo2Speaker dials the robot and returns a ready sink.
func NewGo2Speaker(ctx context.Context, cfg Config, tcfg tts.Config) (*Go2Speaker, error) {
	p, err := NewPlayer(ctx, cfg)
	if err != nil {
		return nil, err
	}
	client := tcfg.Client
	if client == "" {
		client = "RHVoice-client"
	}
	voice := tcfg.Voice
	if voice == "" {
		voice = "SLT"
	}
	return &Go2Speaker{player: p, client: client, voice: voice,
		rate: tcfg.Rate, volume: tcfg.Volume}, nil
}

func (g *Go2Speaker) synth(text string) ([]byte, int, error) {
	cmd := exec.Command(g.client, "-s", g.voice,
		"-r", ftoa(g.rate), "-v", ftoa(g.volume))
	cmd.Stdin = strings.NewReader(text)
	out, err := cmd.Output()
	if err != nil {
		return nil, 0, err
	}
	i := strings.Index(string(out), "data")
	if i < 0 {
		return nil, 24000, nil
	}
	return out[i+8:], 24000, nil // RHVoice: 24 kHz mono s16le
}

func (g *Go2Speaker) Speak(ctx context.Context, text string) (tts.SpeakResult, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return tts.SpeakResult{Completed: true}, nil
	}
	pcm, rate, err := g.synth(text)
	if err != nil {
		return tts.SpeakResult{}, err
	}
	total := time.Duration(len(pcm)/2) * time.Second / time.Duration(rate)
	played, err := g.player.Play(pcm, rate)
	if err != nil {
		return tts.SpeakResult{}, err
	}
	return tts.SpeakResult{
		Completed:   played >= total-30*time.Millisecond,
		Played:      played,
		Synthesized: total,
	}, nil
}

func (g *Go2Speaker) Cancel() { g.player.Cancel() }

func (g *Go2Speaker) Close() { g.player.Close() }

func ftoa(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
