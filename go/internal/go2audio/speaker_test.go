package go2audio

import (
	"math"
	"testing"

	"gopkg.in/hraban/opus.v2"
)

func TestUpsample24kIsCleanDouble(t *testing.T) {
	// 24k → 48k is exactly ×2 sample-and-hold
	pcm := []byte{0x10, 0x00, 0x20, 0x00} // two samples: 16, 32
	out := upsampleTo48k(pcm, 24000)
	if len(out) != 4 || out[0] != 16 || out[1] != 16 || out[2] != 32 || out[3] != 32 {
		t.Fatalf("upsample = %v", out)
	}
}

func TestUpsample16kIsTriple(t *testing.T) {
	pcm := []byte{0x05, 0x00}
	out := upsampleTo48k(pcm, 16000)
	if len(out) != 3 {
		t.Fatalf("16k→48k must triple, got %d", len(out))
	}
}

// A synthesized tone encodes to Opus and decodes back to ~the same tone —
// proves the encoder path the speaker uses actually produces valid frames.
func TestOpusRoundTripTone(t *testing.T) {
	enc, err := opus.NewEncoder(opusRate, 1, opus.AppVoIP)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := opus.NewDecoder(opusRate, 1)
	if err != nil {
		t.Fatal(err)
	}
	// 440 Hz, one 20 ms frame
	frame := make([]int16, frameSmpls)
	for i := range frame {
		frame[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/opusRate))
	}
	buf := make([]byte, 4000)
	nb, err := enc.Encode(frame, buf)
	if err != nil || nb == 0 {
		t.Fatalf("encode: nb=%d err=%v", nb, err)
	}
	out := make([]int16, frameSmpls)
	ns, err := dec.Decode(buf[:nb], out)
	if err != nil || ns != frameSmpls {
		t.Fatalf("decode: ns=%d err=%v", ns, err)
	}
	// energy preserved (Opus is lossy but a pure tone survives well)
	var e float64
	for _, s := range out {
		e += float64(s) * float64(s)
	}
	if e < 1e6 {
		t.Fatalf("decoded frame is near-silent (e=%.0f)", e)
	}
}
