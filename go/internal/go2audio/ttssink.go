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
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/tts"
)

var _ tts.Speaker = (*Go2Speaker)(nil)

type Go2Speaker struct {
	ctx    context.Context
	cfg    Config
	player *Player
	client string // RHVoice-client
	voice  string
	rate   float64
	volume float64
	mu     sync.Mutex // guards player across reconnects
	closed bool
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
	g := &Go2Speaker{ctx: ctx, cfg: cfg, player: p, client: client,
		voice: voice, rate: tcfg.Rate, volume: tcfg.Volume}
	go g.monitor() // heal the connection even while idle
	return g, nil
}

// monitor watches the live connection and redials with capped backoff the
// moment it drops — recovery must not wait for the next utterance (robot
// reboot / wifi blip / slot loss). Runs until Close.
func (g *Go2Speaker) monitor() {
	for {
		g.mu.Lock()
		p, closed := g.player, g.closed
		g.mu.Unlock()
		if closed {
			return
		}
		select {
		case <-g.ctx.Done():
			return
		case <-p.conn.Dead():
			log.Printf("[go2audio] connection dropped — reconnecting")
			g.reconnect()
		}
	}
}

// reconnect dials a fresh Player and swaps it in under the lock. Backoff is
// capped; it loops until success, Close, or ctx cancellation. Safe to call
// from Speak or the monitor — a redundant call just re-dials a healthy link
// once and returns.
func (g *Go2Speaker) reconnect() {
	backoff := 500 * time.Millisecond
	for {
		g.mu.Lock()
		if g.closed {
			g.mu.Unlock()
			return
		}
		if g.player.conn.Alive() {
			g.mu.Unlock()
			return // someone already healed it
		}
		g.mu.Unlock()

		select {
		case <-g.ctx.Done():
			return
		case <-time.After(backoff):
		}
		np, err := NewPlayer(g.ctx, g.cfg)
		if err != nil {
			log.Printf("[go2audio] reconnect failed (%v) — retry in %s",
				err, backoff)
			if backoff < 8*time.Second {
				backoff *= 2
			}
			continue
		}
		g.mu.Lock()
		old := g.player
		g.player = np
		g.mu.Unlock()
		old.Close() // free the old (already-dead) slot cleanly
		log.Printf("[go2audio] reconnected: %s", g.cfg.IP)
		return
	}
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

	g.mu.Lock()
	p := g.player
	alive := p.conn.Alive()
	g.mu.Unlock()
	if !alive {
		g.reconnect() // block this utterance until the link is back
		g.mu.Lock()
		p = g.player
		g.mu.Unlock()
	}
	played, err := p.Play(pcm, rate)
	if err != nil {
		return tts.SpeakResult{}, err
	}
	return tts.SpeakResult{
		Completed:   played >= total-30*time.Millisecond,
		Played:      played,
		Synthesized: total,
	}, nil
}

func (g *Go2Speaker) Cancel() {
	g.mu.Lock()
	p := g.player
	g.mu.Unlock()
	p.Cancel()
}

func (g *Go2Speaker) Close() {
	g.mu.Lock()
	g.closed = true
	p := g.player
	g.mu.Unlock()
	p.Close()
}

func ftoa(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
