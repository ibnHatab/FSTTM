// Package audio captures 16 kHz mono s16le microphone frames via miniaudio
// (malgo) — PulseAudio backend, so the same routing (default source, AEC
// virtual mics) works as in the Python engine. miniaudio resamples in native
// code; frames are delivered by the device callback, so the process is fully
// idle between buffers.
package audio

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/gen2brain/malgo"

	"github.com/ibnHatab/fsttm/go/internal/vad"
)

// Capture streams vad.FrameBytes-sized frames into `frames` until ctx is
// done. Blocking send is deliberate: if the consumer stalls, backpressure
// drops audio at the miniaudio layer rather than growing memory.
// Device selects the capture device: Name (a substring of the device's name or ALSA id,
// e.g. "ReSpeaker" or "hw:1,0"; "" the default), opened with Channels channels of which
// Channel is kept (the ReSpeaker read straight from ALSA: 6 and 0, its processed speech).
type Device struct {
	Name     string
	Channels int
	Channel  int
	ALSA     bool // the ALSA backend only (a device PipeWire must not touch)
}

// Capture is CaptureFrom with the default device.
func Capture(ctx context.Context, frames chan<- []byte) error {
	return CaptureFrom(ctx, Device{}, frames)
}

// CaptureFrom streams 20 ms mono S16 frames at 16 kHz from the device.
func CaptureFrom(ctx context.Context, d Device, frames chan<- []byte) error {
	backends := []malgo.Backend{malgo.BackendPulseaudio, malgo.BackendAlsa}
	if d.ALSA {
		backends = []malgo.Backend{malgo.BackendAlsa}
	}
	mctx, err := malgo.InitContext(backends, malgo.ContextConfig{}, func(string) {})
	if err != nil {
		return fmt.Errorf("audio: init context: %w", err)
	}
	defer func() {
		_ = mctx.Uninit()
		mctx.Free()
	}()

	cfg := malgo.DefaultDeviceConfig(malgo.Capture)
	cfg.Capture.Format = malgo.FormatS16
	ch := d.Channels
	if ch < 1 {
		ch = 1
	}
	cfg.Capture.Channels = uint32(ch)
	cfg.SampleRate = vad.SampleRate
	if d.Name != "" {
		infos, err := mctx.Devices(malgo.Capture)
		if err != nil {
			return fmt.Errorf("audio: devices: %w", err)
		}
		found := false
		for _, inf := range infos {
			if strings.Contains(strings.ToLower(inf.Name()), strings.ToLower(d.Name)) ||
				strings.Contains(inf.ID.String(), d.Name) {
				id := inf.ID
				cfg.Capture.DeviceID = id.Pointer()
				log.Printf("[audio] capture %q (%d ch, keeping ch %d)", inf.Name(), ch, d.Channel)
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("audio: no capture device like %q", d.Name)
		}
	}

	var pending []byte
	onRecv := func(_, in []byte, _ uint32) {
		if ch > 1 {
			in = PickChannel(in, ch, d.Channel)
		}
		pending = append(pending, in...)
		for len(pending) >= vad.FrameBytes {
			frame := make([]byte, vad.FrameBytes)
			copy(frame, pending[:vad.FrameBytes])
			pending = pending[vad.FrameBytes:]
			select {
			case frames <- frame:
			case <-ctx.Done():
				return
			}
		}
	}

	dev, err := malgo.InitDevice(mctx.Context, cfg, malgo.DeviceCallbacks{Data: onRecv})
	if err != nil {
		return fmt.Errorf("audio: init device: %w", err)
	}
	defer dev.Uninit()

	if err := dev.Start(); err != nil {
		return fmt.Errorf("audio: start: %w", err)
	}
	<-ctx.Done()
	return nil
}

// PickChannel keeps one channel of interleaved S16 frames.
func PickChannel(in []byte, channels, keep int) []byte {
	n := len(in) / (2 * channels)
	out := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		o := (i*channels + keep) * 2
		out[2*i], out[2*i+1] = in[o], in[o+1]
	}
	return out
}
