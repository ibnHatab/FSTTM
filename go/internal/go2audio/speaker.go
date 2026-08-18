package go2audio

// Speaker — an fsttm/tts sink that streams synthesized PCM to the Go2 over
// WebRTC instead of a local ALSA device. RHVoice emits 24 kHz mono s16le;
// WebRTC/Opus wants 48 kHz, so we upsample ×2 (linear) and Opus-encode in
// 20 ms frames. Cancel stops the stream mid-utterance (the barge-in path).

import (
	"context"
	"fmt"
	"sync"
	"time"

	"gopkg.in/hraban/opus.v2"
)

const (
	opusRate    = 48000
	frameMs     = 20
	frameSmpls  = opusRate / 1000 * frameMs // 960 samples/frame @48k mono
)

// Player streams PCM to a Go2 speaker; satisfies the shape the TTS driver's
// PcmSink expects (Play blocks until done or cancelled).
type Player struct {
	conn *Conn
	enc  *opus.Encoder
	mu   sync.Mutex
	stop chan struct{}
}

func NewPlayer(ctx context.Context, cfg Config) (*Player, error) {
	conn, err := Connect(ctx, cfg)
	if err != nil {
		return nil, err
	}
	enc, err := opus.NewEncoder(opusRate, 1, opus.AppVoIP)
	if err != nil {
		conn.Close()
		return nil, err
	}
	return &Player{conn: conn, enc: enc, stop: make(chan struct{})}, nil
}

// Play streams one utterance (16-bit mono PCM at inRate Hz). Returns
// (played, error): played is how much audio actually reached the robot,
// short of the whole clip when cancelled.
func (p *Player) Play(pcm []byte, inRate int) (time.Duration, error) {
	samples := upsampleTo48k(pcm, inRate)
	frame := make([]int16, frameSmpls)
	buf := make([]byte, 4000)
	var played time.Duration
	tick := time.NewTicker(frameMs * time.Millisecond)
	defer tick.Stop()

	for off := 0; off < len(samples); off += frameSmpls {
		select {
		case <-p.stop:
			return played, nil // cancelled (barge-in)
		case <-tick.C:
		}
		n := copy(frame, samples[off:])
		for i := n; i < frameSmpls; i++ {
			frame[i] = 0
		}
		nb, err := p.enc.Encode(frame, buf)
		if err != nil {
			return played, fmt.Errorf("go2audio: opus encode: %w", err)
		}
		if err := p.conn.WriteOpus(buf[:nb], frameMs*time.Millisecond); err != nil {
			return played, err
		}
		played += frameMs * time.Millisecond
	}
	return played, nil
}

// Cancel stops the in-flight Play within one frame.
func (p *Player) Cancel() {
	p.mu.Lock()
	defer p.mu.Unlock()
	select {
	case <-p.stop:
	default:
		close(p.stop)
	}
	p.stop = make(chan struct{})
}

func (p *Player) Close() { p.conn.Close() }

// upsampleTo48k: s16le mono at inRate → []int16 at 48 kHz (integer ratio for
// 24k ×2 / 16k ×3; linear otherwise). RHVoice is 24k so this is a clean ×2.
func upsampleTo48k(pcm []byte, inRate int) []int16 {
	n := len(pcm) / 2
	in := make([]int16, n)
	for i := 0; i < n; i++ {
		in[i] = int16(uint16(pcm[2*i]) | uint16(pcm[2*i+1])<<8)
	}
	if opusRate%inRate == 0 {
		r := opusRate / inRate
		out := make([]int16, n*r)
		for i := 0; i < n; i++ {
			for k := 0; k < r; k++ {
				out[i*r+k] = in[i]
			}
		}
		return out
	}
	out := make([]int16, n*opusRate/inRate)
	for j := range out {
		src := float64(j) * float64(inRate) / float64(opusRate)
		i0 := int(src)
		if i0 >= n-1 {
			out[j] = in[n-1]
			continue
		}
		f := src - float64(i0)
		out[j] = int16(float64(in[i0])*(1-f) + float64(in[i0+1])*f)
	}
	return out
}

var _ = context.Background
