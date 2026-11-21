package tts

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Piper speaks through a Piper HTTP server (python -m piper.http_server: the voice loaded
// once, a WAV per request) and plays the WAV with a player (paplay by default, so the
// sound goes through PipeWire to the default sink, the ReSpeaker's jack on the NUC, which
// its echo canceller hears). Cancel kills the player; the fraction heard is the time
// played over the WAV's length.
type Piper struct {
	URL  string // http://127.0.0.1:5150
	HTTP *http.Client
	wavPlayer
}

func NewPiper(url, sink string) *Piper {
	if url == "" {
		url = "http://127.0.0.1:5150"
	}
	return &Piper{URL: strings.TrimRight(url, "/"), HTTP: &http.Client{Timeout: 20 * time.Second}, wavPlayer: newWavPlayer(sink)}
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
	gen := p.begin()
	wav, err := p.Synthesize(ctx, text)
	if err != nil {
		return SpeakResult{}, err
	}
	return p.play(ctx, wav, gen)
}
