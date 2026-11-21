package tts

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Piper speaks through a Piper HTTP server (python -m piper.http_server: the voice loaded
// once, a WAV per request) and plays the WAV with a player (paplay by default, so the
// sound goes through PipeWire to the default sink, the ReSpeaker's jack on the NUC, which
// its echo canceller hears). Cancel kills the player; the fraction heard is the time
// played over the WAV's length.
type Piper struct {
	URL    string   // http://127.0.0.1:5150
	Player []string // the playback command, a WAV file appended (default paplay [--device=Sink])
	HTTP   *http.Client
	mu     sync.Mutex
	cur    *exec.Cmd
}

func NewPiper(url, sink string) *Piper {
	if url == "" {
		url = "http://127.0.0.1:5150"
	}
	player := []string{"paplay"}
	if sink != "" {
		player = append(player, "--device="+sink)
	}
	return &Piper{URL: strings.TrimRight(url, "/"), Player: player, HTTP: &http.Client{Timeout: 20 * time.Second}}
}

// Synthesize is one WAV for the text.
func (p *Piper) Synthesize(ctx context.Context, text string) ([]byte, error) {
	body, _ := json.Marshal(map[string]string{"text": text})
	req, err := http.NewRequestWithContext(ctx, "POST", p.URL+"/synthesize", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("piper: %w", err)
	}
	defer resp.Body.Close()
	wav, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("piper: HTTP %d: %.200s", resp.StatusCode, wav)
	}
	return wav, nil
}

// WavDuration is a PCM WAV's length (0 when the header is not one).
func WavDuration(wav []byte) time.Duration {
	if len(wav) < 44 || string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		return 0
	}
	var byteRate uint32
	for off := 12; off+8 <= len(wav); {
		id, size := string(wav[off:off+4]), int(binary.LittleEndian.Uint32(wav[off+4:off+8]))
		if id == "fmt " && off+16 <= len(wav) {
			byteRate = binary.LittleEndian.Uint32(wav[off+16 : off+20])
		}
		if id == "data" && byteRate > 0 {
			if off+8+size > len(wav) {
				size = len(wav) - off - 8
			}
			return time.Duration(float64(size) / float64(byteRate) * float64(time.Second))
		}
		off += 8 + size + size%2
	}
	return 0
}

func (p *Piper) Speak(ctx context.Context, text string) (SpeakResult, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return SpeakResult{Completed: true}, nil
	}
	wav, err := p.Synthesize(ctx, text)
	if err != nil {
		return SpeakResult{}, err
	}
	total := WavDuration(wav)
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

func (p *Piper) Cancel() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cur != nil && p.cur.Process != nil {
		_ = syscall.Kill(-p.cur.Process.Pid, syscall.SIGKILL)
		p.cur = nil
	}
}
