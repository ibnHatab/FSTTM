// paradee-say: text → the Paradee voice as a WAV (and its phonemes and timing), for checks
// against the Python Paradee.
//
//	paradee-say [-dir models/paradee] [-voice af_heart] [-speed 1] [-o out.wav] [-ps] TEXT...
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/paradee"
	"github.com/ibnHatab/fsttm/go/internal/tts"
)

func main() {
	dir := flag.String("dir", "models/paradee", "the model and lexicons")
	voice := flag.String("voice", "af_heart", "voice")
	speed := flag.Float64("speed", 1, "speed")
	threads := flag.Int("threads", 1, "onnxruntime threads")
	out := flag.String("o", "", "write a WAV here")
	ps := flag.Bool("ps", false, "print the phonemes only")
	flag.Parse()
	text := strings.Join(flag.Args(), " ")
	t0 := time.Now()
	m, err := paradee.Open(*dir, *threads)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	load := time.Since(t0)
	if *ps {
		for _, c := range m.Chunks(text) {
			fmt.Println(c)
		}
		return
	}
	t1 := time.Now()
	a, err := m.Synthesize(text, float32(*speed), *voice)
	if err != nil {
		log.Fatal(err)
	}
	synth := time.Since(t1)
	dur := float64(len(a)) / paradee.SampleRate
	fmt.Printf("load %.2fs, synthesis %.2fs for %.2fs of speech (%.1fx real time)\n", load.Seconds(), synth.Seconds(), dur, dur/synth.Seconds())
	if *out != "" {
		if err := os.WriteFile(*out, tts.WAV16(a, paradee.SampleRate), 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
