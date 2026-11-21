package audio

import (
	"encoding/binary"
	"testing"
)

func TestPickChannel(t *testing.T) {
	in := make([]byte, 2*12) // two frames, six channels: values 0..11
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint16(in[2*i:], uint16(i))
	}
	out := PickChannel(in, 6, 0)
	if len(out) != 4 || binary.LittleEndian.Uint16(out) != 0 || binary.LittleEndian.Uint16(out[2:]) != 6 {
		t.Fatalf("% x", out)
	}
}
