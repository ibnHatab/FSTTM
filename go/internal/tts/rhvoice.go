// Package tts speaks through Linux RHVoice — the statistical, CPU-cheap
// engine the Orin deployment uses. Per utterance:
//
//	echo text | RHVoice-client -s SLT -r 0.3 -v -0.1 | aplay -q
//
// Two subprocesses that exist only WHILE speaking: idle cost is exactly
// zero. Barge-in kills the process group, which stops audio instantly
// (same semantics as the historical xsel-to-speech killall).
package tts

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"syscall"
)

type Config struct {
	Client string  // RHVoice-client binary (default "RHVoice-client")
	Voice  string  // -s (default SLT)
	Rate   float64 // -r, -1..1
	Volume float64 // -v, -1..1
	Player string  // playback command (default "aplay -q")
}

type TTS struct {
	cfg Config
	mu  sync.Mutex
	cur *exec.Cmd // in-flight pipeline (client|player), nil when silent
}

func New(cfg Config) (*TTS, error) {
	if cfg.Client == "" {
		cfg.Client = "RHVoice-client"
	}
	if cfg.Voice == "" {
		cfg.Voice = "SLT"
	}
	if cfg.Player == "" {
		cfg.Player = "aplay -q"
	}
	if _, err := exec.LookPath(cfg.Client); err != nil {
		return nil, fmt.Errorf("tts: %s not found (install RHVoice)", cfg.Client)
	}
	return &TTS{cfg: cfg}, nil
}

// Speak synthesizes and plays text, blocking until playback finishes or
// Cancel kills it. Returns whether playback completed uncancelled.
func (t *TTS) Speak(ctx context.Context, text string) (bool, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return true, nil
	}
	playerArgv := strings.Fields(t.cfg.Player)
	// One shell-free process group: client | player
	pipeline := exec.CommandContext(ctx, "sh", "-c",
		fmt.Sprintf("%s -s %s -r %g -v %g | %s",
			t.cfg.Client, t.cfg.Voice, t.cfg.Rate, t.cfg.Volume,
			strings.Join(playerArgv, " ")))
	pipeline.Stdin = strings.NewReader(text)
	pipeline.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	t.mu.Lock()
	if err := pipeline.Start(); err != nil {
		t.mu.Unlock()
		return false, err
	}
	t.cur = pipeline
	t.mu.Unlock()

	err := pipeline.Wait()

	t.mu.Lock()
	cancelled := t.cur == nil // Cancel() ran while we played
	t.cur = nil
	t.mu.Unlock()

	if cancelled {
		return false, nil
	}
	return err == nil, err
}

// Cancel stops the current playback immediately (barge-in). No-op if silent.
func (t *TTS) Cancel() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.cur != nil && t.cur.Process != nil {
		// negative pid → the whole process group (client + player)
		_ = syscall.Kill(-t.cur.Process.Pid, syscall.SIGKILL)
		t.cur = nil
	}
}
