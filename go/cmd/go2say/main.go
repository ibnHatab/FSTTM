// go2say — speak text (RHVoice) or a WAV to a Unitree Go2 speaker over
// WebRTC. Standalone probe for the streaming path; the engine uses the same
// go2audio.Player as a TTS sink.
//
//	go2say -ip 192.168.123.161 -aes <32hex> "Hello from Nina."
//	go2say -ip 192.168.123.161 -wav clip.wav
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/go2audio"
)

func rhvoicePCM(text string) ([]byte, int) {
	cmd := exec.Command("RHVoice-client", "-s", "SLT", "-r", "0.3", "-v", "-0.1")
	cmd.Stdin = strings.NewReader(text)
	out, err := cmd.Output()
	if err != nil {
		log.Fatalf("RHVoice: %v", err)
	}
	i := strings.Index(string(out), "data")
	return out[i+8:], 24000
}

func readWav(path string) ([]byte, int) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	rate := int(binary.LittleEndian.Uint32(b[24:28]))
	i := strings.Index(string(b), "data")
	return b[i+8:], rate
}

func main() {
	ip := flag.String("ip", "", "robot LAN IP")
	aes := flag.String("aes", "", "per-device AES-128 key (32 hex)")
	wav := flag.String("wav", "", "WAV file instead of TTS")
	flag.Parse()
	if *ip == "" {
		log.Fatal("need -ip")
	}

	ctx := context.Background()
	p, err := go2audio.NewPlayer(ctx, go2audio.Config{IP: *ip, AES128: *aes})
	if err != nil {
		log.Fatal(err)
	}
	defer p.Close()

	var pcm []byte
	var rate int
	if *wav != "" {
		pcm, rate = readWav(*wav)
	} else {
		pcm, rate = rhvoicePCM(strings.Join(flag.Args(), " "))
	}
	t0 := time.Now()
	played, err := p.Play(pcm, rate)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("streamed %.1fs in %.1fs", played.Seconds(), time.Since(t0).Seconds())
}
