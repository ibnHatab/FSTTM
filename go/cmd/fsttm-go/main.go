// fsttm-go — the FSTTM spoken-dialog engine in Go: message-passing pipeline
// over whisper.cpp (official Go binding), llama.cpp (thin cgo shim, two-pass
// grammar intents with KV-prefix reuse) and Linux RHVoice TTS.
//
//	fsttm-go -config config.dog.yaml            # voice pipeline
//	fsttm-go -config config.dog.yaml -headless  # stdin → intent → voice
//
// Designed for the Orin: idle means IDLE — every goroutine blocks on a
// channel; a silent room costs one native VAD call per 20 ms frame and the
// GPU runs no kernels between utterances.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"gopkg.in/yaml.v3"

	"github.com/ibnHatab/fsttm/go/internal/audio"
	"github.com/ibnHatab/fsttm/go/internal/llm"
	"github.com/ibnHatab/fsttm/go/internal/pipeline"
	"github.com/ibnHatab/fsttm/go/internal/stt"
	"github.com/ibnHatab/fsttm/go/internal/tts"
	"github.com/ibnHatab/fsttm/go/internal/vad"
)

type config struct {
	LLM struct {
		Model      string `yaml:"model"`
		NCtx       int    `yaml:"n_ctx"`
		NBatch     int    `yaml:"n_batch"`
		NThreads   int    `yaml:"n_threads"`
		NGpuLayers int    `yaml:"n_gpu_layers"`
	} `yaml:"llm"`
	STT struct {
		Model     string   `yaml:"model"`
		Language  string   `yaml:"language"`
		Parasites []string `yaml:"parasites"`
	} `yaml:"stt"`
	TTS struct {
		Engine string  `yaml:"engine"` // librhvoice (default) | subprocess
		Voice  string  `yaml:"voice"`
		Rate   float64 `yaml:"rate"`
		Volume float64 `yaml:"volume"`
		Player string  `yaml:"player"`
	} `yaml:"tts"`
	VAD struct {
		Aggressiveness int `yaml:"aggressiveness"`
		PaddingMs      int `yaml:"padding_ms"`
	} `yaml:"vad"`
	Prompt   string `yaml:"prompt"`    // system prompt file
	Grammar  string `yaml:"grammar"`   // GBNF file
	WakeWord string `yaml:"wake_word"` // "" → always awake
	BargeIn  bool   `yaml:"barge_in"`  // needs AEC in the audio path
}

func main() {
	cfgPath := flag.String("config", "config.dog.yaml", "YAML config")
	headless := flag.Bool("headless", false, "stdin text instead of the mic")
	flag.Parse()
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	raw, err := os.ReadFile(*cfgPath)
	if err != nil {
		log.Fatal(err)
	}
	var cfg config
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		log.Fatal(err)
	}
	prompt, err := os.ReadFile(cfg.Prompt)
	if err != nil {
		log.Fatal(err)
	}
	gbnf, err := os.ReadFile(cfg.Grammar)
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
	defer stop()

	// ── drivers ──────────────────────────────────────────────────────────
	l, err := llm.Load(llm.Config{ModelPath: cfg.LLM.Model, NCtx: cfg.LLM.NCtx,
		NBatch: cfg.LLM.NBatch, NThreads: cfg.LLM.NThreads,
		NGpuLayers: cfg.LLM.NGpuLayers})
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	if n, err := l.PrimePrefix(string(prompt)); err != nil {
		log.Fatal(err)
	} else {
		log.Printf("[llm] intent prefix pre-warmed: %d tok", n)
	}

	s, err := stt.New(cfg.STT.Model, cfg.STT.Language, cfg.STT.Parasites)
	if err != nil {
		log.Fatal(err)
	}
	defer s.Close()
	log.Print("[stt] whisper ready (warmed)")

	t, err := tts.NewSpeaker(tts.Config{Engine: cfg.TTS.Engine,
		Voice: cfg.TTS.Voice, Rate: cfg.TTS.Rate,
		Volume: cfg.TTS.Volume, Player: cfg.TTS.Player})
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("[tts] rhvoice ready (engine=%T)", t)

	eng := pipeline.New(pipeline.Config{
		SystemPrompt: string(prompt), GBNF: string(gbnf),
		WakeWord: cfg.WakeWord, BargeIn: cfg.BargeIn,
	}, l, s, t)

	// SIGUSR1 → system-initiated announcement (transition 5: the system may
	// freely take an unclaimed floor). Robot processes use Engine.Announce
	// directly; the signal gives shells/tests a live trigger:
	//   kill -USR1 $(pidof fsttm-go)
	usr1 := make(chan os.Signal, 1)
	signal.Notify(usr1, syscall.SIGUSR1)
	go func() {
		for range usr1 {
			eng.Announce("System check. All services nominal.")
		}
	}()

	// ── sources ──────────────────────────────────────────────────────────
	var vadEvents chan vad.Event
	var textIn chan string

	if *headless {
		textIn = make(chan string)
		go func() {
			defer close(textIn)
			sc := bufio.NewScanner(os.Stdin)
			fmt.Print("> ")
			for sc.Scan() {
				line := sc.Text()
				if line != "" {
					textIn <- line
				}
				fmt.Print("> ")
			}
		}()
	} else {
		frames := make(chan []byte, 8)
		vadEvents = make(chan vad.Event, 4)
		seg, err := vad.New(cfg.VAD.Aggressiveness, cfg.VAD.PaddingMs)
		if err != nil {
			log.Fatal(err)
		}
		go func() {
			if err := audio.Capture(ctx, frames); err != nil {
				log.Printf("[audio] %v", err)
				stop()
			}
		}()
		go seg.Run(ctx, frames, vadEvents)
		log.Printf("[vad] streaming (aggr=%d padding=%dms) — say something",
			cfg.VAD.Aggressiveness, cfg.VAD.PaddingMs)
	}

	eng.Run(ctx, vadEvents, textIn)
	fmt.Println("\nbye.")
}
