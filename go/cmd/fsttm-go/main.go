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
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ibnHatab/fsttm/go/internal/aec"
	"github.com/ibnHatab/fsttm/go/internal/audio"
	"github.com/ibnHatab/fsttm/go/internal/cloud"
	"github.com/ibnHatab/fsttm/go/internal/fsm"
	"github.com/ibnHatab/fsttm/go/internal/go2audio"
	"github.com/ibnHatab/fsttm/go/internal/llm"
	"github.com/ibnHatab/fsttm/go/internal/nina"
	"github.com/ibnHatab/fsttm/go/internal/panel"
	"github.com/ibnHatab/fsttm/go/internal/pipeline"
	"github.com/ibnHatab/fsttm/go/internal/stt"
	"github.com/ibnHatab/fsttm/go/internal/tts"
	"github.com/ibnHatab/fsttm/go/internal/vad"
	"github.com/ibnHatab/fsttm/go/internal/voiceid"
	"github.com/ibnHatab/fsttm/go/pkg/hometools"
	"github.com/ibnHatab/fsttm/go/pkg/respeaker"
	"github.com/ibnHatab/fsttm/go/pkg/streams"
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
		Model      string   `yaml:"model"`
		Language   string   `yaml:"language"`
		Parasites  []string `yaml:"parasites"`
		AudioCtx   uint     `yaml:"audio_ctx"` // 512: short utterances ~1.6x faster (0: whisper's 1500)
		Threads    uint     `yaml:"threads"`
		NoFallback bool     `yaml:"no_fallback"`
	} `yaml:"stt"`
	// Cloud: the LLM over an OpenAI-compatible API (Gemini, OpenRouter, a llama-server):
	// free conversation, the streams in the prompt, the home's tools. Set → llm: unused.
	Cloud struct {
		BaseURL      string         `yaml:"base_url"`
		Model        string         `yaml:"model"`
		APIKeyFile   string         `yaml:"api_key_file"`
		APIKeyEnv    string         `yaml:"api_key_env"`
		Persona      string         `yaml:"persona"`
		Temperature  float64        `yaml:"temperature"`
		MaxTokens    int            `yaml:"max_tokens"`
		TimeoutS     float64        `yaml:"timeout_s"`
		RequestExtra map[string]any `yaml:"request_extra"`
	} `yaml:"cloud"`
	Streams struct {
		Addr string `yaml:"addr"` // the door the env / reasoning streams come in by (127.0.0.1:8766)
	} `yaml:"streams"`
	Tools struct {
		Addr string `yaml:"addr"` // the rule engine's tool port (127.0.0.1:8768)
	} `yaml:"tools"`
	Panel struct {
		Addr string `yaml:"addr"` // the page: 0.0.0.0:8767
		// the page's mic and voice switches, kept across restarts in this file ("": the
		// default, $XDG_STATE_HOME/fsttm/switches.json; "-": not kept, both on at start)
		SwitchesFile string `yaml:"switches_file"`
	} `yaml:"panel"`
	Audio struct {
		Device   string `yaml:"device"`   // a capture device by name ("ReSpeaker"); "" the default
		Channels int    `yaml:"channels"` // open it with this many (the ReSpeaker: 6)
		Channel  int    `yaml:"channel"`  // and keep this one (the ReSpeaker: 0, its processed speech)
		ALSA     bool   `yaml:"alsa"`     // straight from ALSA (PipeWire's capture of the array is broken)
	} `yaml:"audio"`
	// Respeaker: the ReSpeaker 4 Mic Array's XVF-3000 parameters set at start (go/pkg/respeaker; they
	// live in the array's RAM, so a replug resets them: restart the engine after one)
	Respeaker map[string]float64 `yaml:"respeaker"`
	TTS       struct {
		Engine string  `yaml:"engine"` // librhvoice (default) | subprocess | go2 | piper | paradee
		Voice  string  `yaml:"voice"`
		Rate   float64 `yaml:"rate"`
		Volume float64 `yaml:"volume"`
		Player string  `yaml:"player"`
		Piper  string  `yaml:"piper"` // engine piper: the Piper HTTP server (http://127.0.0.1:5150)
		Sink   string  `yaml:"sink"`  // engine piper / paradee: the PipeWire sink ("" the default)
		// engine paradee: the model and lexicons (models/paradee), its voice (af_heart), speed
		// (1.0; the narration script uses 1.3) and onnxruntime threads
		Paradee struct {
			Dir     string  `yaml:"dir"`
			Speed   float64 `yaml:"speed"`
			Threads int     `yaml:"threads"`
		} `yaml:"paradee"`
	} `yaml:"tts"`
	Go2 struct { // tts.engine: go2 — stream TTS to the robot speaker (WebRTC)
		IP     string `yaml:"ip"`
		AES128 string `yaml:"aes_128_key"`
	} `yaml:"go2_speaker"`
	VAD struct {
		Aggressiveness int     `yaml:"aggressiveness"`
		PaddingMs      int     `yaml:"padding_ms"`
		MaxUtteranceS  float64 `yaml:"max_utterance_s"` // cut a longer utterance (media in the room never pauses); 0: no bound
	} `yaml:"vad"`
	AEC struct {
		Enabled      bool   `yaml:"enabled"`
		RNNoise      bool   `yaml:"rnnoise"`
		Method       string `yaml:"method"`
		SourceMaster string `yaml:"source_master"`
	} `yaml:"aec"`
	VoiceID struct {
		Model     string  `yaml:"model"`     // speaker-embedding ONNX
		Profile   string  `yaml:"profile"`   // owner imprint (fsttm-imprint)
		Threshold float64 `yaml:"threshold"` // cosine acceptance (0.40)
		Gate      string  `yaml:"gate"`      // off | wake | always
	} `yaml:"voice_id"`
	Attention struct {
		Enabled      bool     `yaml:"enabled"`
		WakeWords    []string `yaml:"wake_words"`
		SleepPhrases []string `yaml:"sleep_phrases"`
		AwakeS       float64  `yaml:"awake_s"` // back to sleep after this many quiet seconds (0: never)
	} `yaml:"attention"`
	Nina struct {
		Enabled     bool   `yaml:"enabled"`
		Transport   string `yaml:"transport"`    // jsonl (SIL) | ros (rclgo native)
		CanvasPack  string `yaml:"canvas_pack"`  // canvas_query_pack.npz
		PhraseCache string `yaml:"phrase_cache"` // phrase_cache.npz
	} `yaml:"nina"`
	Prompt  string `yaml:"prompt"`   // system prompt file
	Grammar string `yaml:"grammar"`  // GBNF file
	BargeIn string `yaml:"barge_in"` // off | vad | confirm
}

func main() {
	cfgPath := flag.String("config", "config.dog.yaml", "YAML config")
	headless := flag.Bool("headless", false, "stdin text instead of the mic")
	wavIn := flag.String("transcribe", "", "transcribe a 16 kHz S16 WAV (its channel audio.channel) with the stt: settings, print the time and the text, exit")
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
	if e := os.Getenv("FSTTM_TTS_ENGINE"); e != "" { // the unit's choice over the file's (deploy/nuc: the swap)
		cfg.TTS.Engine = e
	}
	if *wavIn != "" {
		transcribeFile(cfg, *wavIn)
		return
	}
	useCloud := cfg.Cloud.Model != ""
	var prompt, gbnf []byte
	if !useCloud { // the local intent LLM: its prompt and grammar
		if prompt, err = os.ReadFile(cfg.Prompt); err != nil {
			log.Fatal(err)
		}
		if gbnf, err = os.ReadFile(cfg.Grammar); err != nil {
			log.Fatal(err)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
	defer stop()

	// ── echo cancellation + noise suppression ────────────────────────────
	// module-echo-cancel (webrtc AEC + NS, optional RNNoise chain for motor
	// noise); routes the PulseAudio defaults so capture/playback pick up the
	// EC devices automatically. Restored on exit.
	if !*headless {
		ec, err := aec.Start(aec.Config{Enabled: cfg.AEC.Enabled,
			RNNoise: cfg.AEC.RNNoise, Method: cfg.AEC.Method,
			SourceMaster: cfg.AEC.SourceMaster})
		if err != nil {
			log.Printf("[aec] %v — continuing on the raw mic", err)
		} else {
			defer ec.Close()
		}
	}

	// ── owner voice imprint ──────────────────────────────────────────────
	var owner pipeline.OwnerVerifier
	if cfg.VoiceID.Model != "" && cfg.VoiceID.Profile != "" {
		v, err := voiceid.New(voiceid.Config{Model: cfg.VoiceID.Model,
			Profile: cfg.VoiceID.Profile, Threshold: cfg.VoiceID.Threshold})
		if err != nil {
			log.Printf("[voiceid] %v — running unimprinted", err)
		} else {
			owner = v
			log.Printf("[voiceid] imprinted to %q (gate=%s)",
				v.Owner(), cfg.VoiceID.Gate)
		}
	}

	// ── drivers ──────────────────────────────────────────────────────────
	var intentLLM pipeline.IntentGen
	if !useCloud {
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
		intentLLM = l
	}

	s, err := stt.New(cfg.STT.Model, cfg.STT.Language, cfg.STT.Parasites)
	if err != nil {
		log.Fatal(err)
	}
	defer s.Close()
	s.AudioCtx, s.Threads = cfg.STT.AudioCtx, cfg.STT.Threads
	s.NoFallback = cfg.STT.NoFallback
	log.Printf("[stt] whisper ready (warmed; audio_ctx=%d)", s.AudioCtx)

	tcfg := tts.Config{Engine: cfg.TTS.Engine, Voice: cfg.TTS.Voice,
		Rate: cfg.TTS.Rate, Volume: cfg.TTS.Volume, Player: cfg.TTS.Player}
	var t tts.Speaker
	if cfg.TTS.Engine == "piper" {
		t = tts.NewPiper(cfg.TTS.Piper, cfg.TTS.Sink)
	} else if cfg.TTS.Engine == "paradee" {
		pc := cfg.TTS.Paradee
		if pc.Dir == "" {
			pc.Dir = "models/paradee"
		}
		pd, err := tts.NewParadee(pc.Dir, cfg.TTS.Voice, pc.Speed, pc.Threads, cfg.TTS.Sink)
		if err != nil {
			log.Fatalf("[tts] paradee: %v", err)
		}
		t = pd
	} else if cfg.TTS.Engine == "go2" {
		// Stream TTS to the Go2 speaker over WebRTC (no local audio device).
		gs, err := go2audio.NewGo2Speaker(ctx,
			go2audio.Config{IP: cfg.Go2.IP, AES128: cfg.Go2.AES128}, tcfg)
		if err != nil {
			log.Fatalf("[tts] go2 speaker: %v", err)
		}
		t = gs
	} else {
		var err error
		if t, err = tts.NewSpeaker(tcfg); err != nil {
			log.Fatal(err)
		}
	}
	log.Printf("[tts] speaker ready (engine=%T)", t)

	eng := pipeline.New(pipeline.Config{
		SystemPrompt: string(prompt), GBNF: string(gbnf),
		Attention:    cfg.Attention.Enabled,
		WakeWords:    cfg.Attention.WakeWords,
		SleepPhrases: cfg.Attention.SleepPhrases,
		BargeIn:      cfg.BargeIn,
		OwnerGate:    cfg.VoiceID.Gate,
	}, intentLLM, s, t)
	eng.Owner = owner
	eng.Attn.AwakeFor = time.Duration(cfg.Attention.AwakeS * float64(time.Second))

	// ── the cloud agent, the streams beside the conversation, the home's tools, the panel ──
	var pan *panel.Server
	if cfg.Panel.Addr != "" {
		pan = panel.New()
		if err := pan.Serve(cfg.Panel.Addr); err != nil {
			log.Fatalf("[panel] %v", err)
		}
	}
	if useCloud {
		ag := &cloud.Agent{
			Client: cloud.NewClient(orDefault(cfg.Cloud.BaseURL, "https://generativelanguage.googleapis.com/v1beta/openai/"),
				cfg.Cloud.Model, cloud.KeyFrom(cfg.Cloud.APIKeyFile, cfg.Cloud.APIKeyEnv), cfg.Cloud.RequestExtra,
				time.Duration(cfg.Cloud.TimeoutS*float64(time.Second))),
			Streams: cloud.NewStreams(), Persona: cfg.Cloud.Persona, Temp: cfg.Cloud.Temperature,
			MaxTokens: cfg.Cloud.MaxTokens,
		}
		if ag.MaxTokens == 0 {
			ag.MaxTokens = 200
		}
		if cfg.Tools.Addr != "" {
			ag.Tools = &hometools.Client{Addr: cfg.Tools.Addr}
		}
		if pan != nil {
			ag.OnTool = pan.Tool
		}
		eng.Chat = ag
		log.Printf("[cloud] %s at %s (tools %s)", ag.Client.Model, ag.Client.URL, orDefault(cfg.Tools.Addr, "none"))
		if cfg.Streams.Addr != "" {
			if _, err := streams.Open(cfg.Streams.Addr, func(ev streams.Event) {
				if ev.Stream == "env" {
					ag.Streams.Env(ev.Key, ev.Text, ev.TS)
					if pan != nil {
						pan.Env(ev.Key, ev.Text, ev.TS)
					}
				} else {
					ag.Streams.Reason(*ev.Text, ev.TS)
					if pan != nil {
						pan.Reasoning(*ev.Text, ev.TS)
					}
				}
			}); err != nil {
				log.Fatalf("[streams] %v", err)
			}
		}
	}
	if pan != nil {
		// the attention as the page shows it: polled (the engine's idle expiry is lazy, so a quiet
		// spell back to ASLEEP has no event of its own); a change is a state frame and an event
		go func() {
			last := ""
			tick := time.NewTicker(500 * time.Millisecond)
			defer tick.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-tick.C:
				}
				st := map[bool]string{true: "AWAKE", false: "ASLEEP"}[eng.Attn.Peek()]
				if !eng.Attn.Enabled {
					st = "OFF"
				}
				if st != last {
					last = st
					pan.State(map[string]any{"attention": st})
					pan.Event("attention", map[string]any{"state": st})
				}
			}
		}()
		// the mic and voice switches: the page's ops to the engine, their state to every page,
		// kept in a small file across restarts
		swFile := switchesFile(cfg.Panel.SwitchesFile)
		if sw, ok := loadSwitches(swFile); ok {
			if !sw.Mic {
				eng.Control("mic off")
			}
			if !sw.Voice {
				eng.Control("voice off")
			}
			log.Printf("[panel] switches from %s: mic %v, voice %v", swFile, sw.Mic, sw.Voice)
		}
		pan.State(map[string]any{"mic": eng.Mic(), "voice": eng.Voice()})
		pan.OnCtl = func(op string) {
			eng.Control(op)
			pan.State(map[string]any{"mic": eng.Mic(), "voice": eng.Voice()})
			saveSwitches(swFile, switches{Mic: eng.Mic(), Voice: eng.Voice()})
			log.Printf("[panel] %s (mic %v, voice %v)", op, eng.Mic(), eng.Voice())
		}
		eng.Typed = make(chan string, 4)
		var typedIDs []string
		var mu sync.Mutex
		pan.OnSay = func(id, text string) {
			mu.Lock()
			typedIDs = append(typedIDs, id)
			mu.Unlock()
			eng.Typed <- text
		}
		cur := ""
		prevChange := eng.Turn.OnChange
		eng.Turn.OnChange = func(event string, src, dst fsm.State) {
			if prevChange != nil {
				prevChange(event, src, dst)
			}
			pan.Event("fsttm", map[string]any{"event": event, "src": string(src), "dst": string(dst)})
		}
		prev := eng.Events
		eng.Events = func(kind string, payload any) {
			if prev != nil {
				prev(kind, payload)
			}
			switch p := payload.(type) {
			case map[string]any:
				pan.Event(kind, p)
			case string:
				pan.Event(kind, map[string]any{"text": p})
			}
			txt, _ := payload.(string)
			switch kind {
			case "typed":
				mu.Lock()
				if len(typedIDs) > 0 {
					cur, typedIDs = typedIDs[0], typedIDs[1:]
				}
				mu.Unlock()
			case "heard":
				cur = ""
				pan.Said("heard", "", txt)
			case "asleep":
				pan.Said("asleep", "", txt)
			case "doubtful":
				pan.Said("doubtful", "", txt)
			case "reply":
				pan.Said("reply", cur, txt)
				cur = ""
			case "dialog":
				pan.State(map[string]any{"dialog": txt, "floor": string(eng.Turn.State),
					"attention": map[bool]string{true: "AWAKE", false: "ASLEEP"}[eng.Attn.Awake()]})
			}
		}
	}

	// ── Nina robot link (nina_speak override) ────────────────────────────
	// jsonl events on stdout → nina_relay → the contracted ROS topics; say
	// requests arrive on stdin. Motion doctrine: see internal/nina.
	if cfg.Nina.Enabled {
		var link nina.RobotLink
		if cfg.Nina.Transport == "ros" {
			rl, err := nina.NewRosLink(ctx, eng.Announce)
			if err != nil {
				log.Fatalf("[nina] %v", err)
			}
			link = rl
		} else {
			link = nina.NewLink(os.Stdout)
		}
		var canvas *nina.Canvas
		if cfg.Nina.CanvasPack != "" {
			c, err := nina.LoadCanvas(cfg.Nina.CanvasPack, cfg.Nina.PhraseCache)
			if err != nil {
				log.Printf("[nina] canvas unavailable: %v — semantic "+
					"queries will find nothing", err)
			} else {
				canvas = c
			}
		}
		eng.Dsp = nina.NewDispatcher(link, canvas)
		eng.Events = func(kind string, payload any) {
			switch kind {
			case "intent":
				if r, ok := payload.(*llm.Result); ok {
					link.Intent(json.RawMessage(r.JSON), r.Voice)
				}
			case "dialog":
				if s, ok := payload.(string); ok {
					link.DialogState(s)
				}
			}
		}
		if jl, ok := link.(*nina.Link); ok && !*headless {
			// jsonl transport: the relay owns our stdin (say requests). In
			// -headless the terminal owns stdin — no say channel then.
			// (ros transport gets /nina/say via its own subscription.)
			go jl.ReadSay(os.Stdin, eng.Announce)
		}
		log.Printf("[nina] robot link active (transport=%s)",
			map[bool]string{true: "ros", false: "jsonl"}[cfg.Nina.Transport == "ros"])
	}

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
		prompt := func() { fmt.Print("> ") }
		if cfg.Nina.Enabled {
			// stdout belongs to jsonl events when the robot link is active
			prompt = func() { fmt.Fprint(os.Stderr, "> ") }
		}
		go func() {
			defer close(textIn)
			sc := bufio.NewScanner(os.Stdin)
			prompt()
			for sc.Scan() {
				line := sc.Text()
				if line != "" {
					textIn <- line
				}
				prompt()
			}
		}()
	} else {
		frames := make(chan []byte, 8)
		vadEvents = make(chan vad.Event, 4)
		seg, err := vad.New(cfg.VAD.Aggressiveness, cfg.VAD.PaddingMs)
		if err != nil {
			log.Fatal(err)
		}
		seg.MaxMs = int(cfg.VAD.MaxUtteranceS * 1000)
		// the mic switch: while off, the capture keeps running and the VAD gets silence (an
		// utterance open when it was switched off closes, and the engine drops it)
		gated := make(chan []byte, 8)
		go func() {
			defer close(gated)
			silence := make([]byte, vad.FrameBytes)
			for f := range frames {
				if !eng.Mic() && len(f) == vad.FrameBytes {
					f = silence
				}
				select {
				case gated <- f:
				case <-ctx.Done():
					return
				}
			}
		}()
		go func() {
			if len(cfg.Respeaker) > 0 {
				if d, err := respeaker.Open(); err != nil {
					log.Printf("[respeaker] not tuned: %v", err)
				} else {
					lines, err := d.Apply(cfg.Respeaker)
					d.Close()
					log.Printf("[respeaker] %s%s", strings.Join(lines, " "), map[bool]string{true: "", false: fmt.Sprintf(" (%v)", err)}[err == nil])
				}
			}
			if err := audio.CaptureFrom(ctx, audio.Device{Name: cfg.Audio.Device,
				Channels: cfg.Audio.Channels, Channel: cfg.Audio.Channel, ALSA: cfg.Audio.ALSA}, frames); err != nil {
				log.Printf("[audio] %v", err)
				stop()
			}
		}()
		go seg.Run(ctx, gated, vadEvents)
		log.Printf("[vad] streaming (aggr=%d padding=%dms max=%.0fs) — say something",
			cfg.VAD.Aggressiveness, cfg.VAD.PaddingMs, cfg.VAD.MaxUtteranceS)
	}

	eng.Run(ctx, vadEvents, textIn)
	fmt.Println("\nbye.")
}

func orDefault(s, d string) string {
	if s == "" {
		return d
	}
	return s
}

// transcribeFile is the -transcribe check: the STT as the engine runs it, on a recording.
func transcribeFile(cfg config, path string) {
	model, lang, parasites, actx, threads, channel := cfg.STT.Model, cfg.STT.Language, cfg.STT.Parasites,
		cfg.STT.AudioCtx, cfg.STT.Threads, cfg.Audio.Channel
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if len(b) < 44 || string(b[0:4]) != "RIFF" {
		log.Fatal("not a WAV")
	}
	ch := int(binary.LittleEndian.Uint16(b[22:24]))
	off := 12
	for off+8 <= len(b) && string(b[off:off+4]) != "data" {
		off += 8 + int(binary.LittleEndian.Uint32(b[off+4:off+8]))
	}
	pcm := b[off+8:]
	if ch > 1 {
		pcm = audio.PickChannel(pcm, ch, channel)
	}
	s, err := stt.New(model, lang, parasites)
	if err != nil {
		log.Fatal(err)
	}
	defer s.Close()
	s.AudioCtx, s.Threads = actx, threads
	s.NoFallback = cfg.STT.NoFallback
	t0 := time.Now()
	r, ok := s.Transcribe(pcm)
	fmt.Printf("%.2fs for %.1fs of audio (audio_ctx %d, window %d): ok=%v redecoded=%v doubtful=%v %q\n", time.Since(t0).Seconds(),
		float64(len(pcm)/2)/16000, actx, stt.CtxFor(actx, len(pcm)/2), ok, r.Redecoded, r.Doubtful, r.Text)
}

// switches are the page's mic and voice, kept across restarts.
type switches struct {
	Mic   bool `json:"mic"`
	Voice bool `json:"voice"`
}

func switchesFile(p string) string {
	if p == "-" {
		return ""
	}
	if p != "" {
		return p
	}
	dir := os.Getenv("XDG_STATE_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".local", "state")
	}
	return filepath.Join(dir, "fsttm", "switches.json")
}

func loadSwitches(path string) (switches, bool) {
	if path == "" {
		return switches{}, false
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return switches{}, false
	}
	sw := switches{Mic: true, Voice: true}
	if json.Unmarshal(b, &sw) != nil {
		return switches{}, false
	}
	return sw, true
}

func saveSwitches(path string, sw switches) {
	if path == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	b, _ := json.Marshal(sw)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Printf("[panel] switches not kept: %v", err)
	}
}
