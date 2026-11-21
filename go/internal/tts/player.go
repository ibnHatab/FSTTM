package tts

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// wavPlayer plays one WAV at a time with a player command (paplay by default: PipeWire, the
// default sink — on the NUC the ReSpeaker's jack, which its echo canceller hears); Cancel
// kills the player, and the fraction heard is the time played over the WAV's length. A
// Cancel that comes while the utterance is still being synthesized (no player yet) counts
// too: begin() takes the cancel generation, and play() of an older one plays nothing.
type wavPlayer struct {
	Player []string
	mu     sync.Mutex
	cur    *exec.Cmd
	gen    uint64
}

// begin: the generation an utterance starts in (before its synthesis).
func (p *wavPlayer) begin() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.gen
}

func newWavPlayer(sink string) wavPlayer {
	player := []string{"paplay"}
	if sink != "" {
		player = append(player, "--device="+sink)
	}
	return wavPlayer{Player: player}
}

func (p *wavPlayer) play(ctx context.Context, wav []byte, gen uint64) (SpeakResult, error) {
	total := WavDuration(wav)
	if ctx.Err() != nil || p.begin() != gen { // cut while it was synthesized
		return SpeakResult{Synthesized: total}, nil
	}
	f, err := os.CreateTemp("", "fsttm-*.wav")
	if err != nil {
		return SpeakResult{}, err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(wav); err != nil {
		f.Close()
		return SpeakResult{}, err
	}
	f.Close()
	cmd := exec.CommandContext(ctx, p.Player[0], append(p.Player[1:], f.Name())...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	p.mu.Lock()
	if p.gen != gen {
		p.mu.Unlock()
		return SpeakResult{Synthesized: total}, nil
	}
	if err := cmd.Start(); err != nil {
		p.mu.Unlock()
		return SpeakResult{}, err
	}
	p.cur = cmd
	p.mu.Unlock()
	t0 := time.Now()
	err = cmd.Wait()
	played := time.Since(t0)
	p.mu.Lock()
	cancelled := p.cur == nil
	p.cur = nil
	p.mu.Unlock()
	if total > 0 && played > total {
		played = total
	}
	res := SpeakResult{Completed: !cancelled && err == nil, Played: played, Synthesized: total}
	if cancelled {
		return res, nil
	}
	return res, err
}

func (p *wavPlayer) Cancel() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gen++
	if p.cur != nil && p.cur.Process != nil {
		_ = syscall.Kill(-p.cur.Process.Pid, syscall.SIGKILL)
		p.cur = nil
	}
}

// WAV16 is float audio as a 16-bit mono WAV.
func WAV16(a []float32, rate int) []byte {
	var b bytes.Buffer
	n := len(a) * 2
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+n))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(rate), uint32(rate * 2), uint16(2), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(n))
	for _, v := range a {
		x := math.Max(-1, math.Min(1, float64(v)))
		binary.Write(&b, binary.LittleEndian, int16(math.Round(x*32767)))
	}
	return b.Bytes()
}
