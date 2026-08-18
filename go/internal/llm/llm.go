// Package llm is a thin cgo shim over the CURRENT llama.cpp C API,
// implementing exactly what fsttm/two_pass.py does: KV-prefix reuse + two-pass
// grammar-constrained intent generation.
//
// Why not go-llama.cpp: it is frozen at llama.cpp March-2024 (no Phi-3
// architecture, pre-sampler-chain API, no llama_memory_* ops) — not on par.
// The modern C API needs only ~a dozen calls, all cgo-friendly.
//
// Build: see go/build.sh — CGO_CFLAGS/-LDFLAGS point at a llama.cpp checkout
// built with -DGGML_CUDA=ON (or CPU-only on the Orin fallback profile).
package llm

/*
#cgo LDFLAGS: -lllama -lggml -lggml-base
#include <stdio.h>
#include <stdlib.h>
#include <llama.h>

// batch helper: single-token decode without repeated cgo slice plumbing
static int decode_one(struct llama_context *ctx, llama_token t) {
    return llama_decode(ctx, llama_batch_get_one(&t, 1));
}

// quiet log sink: drop everything below warnings (llama.cpp logs every
// tensor load at debug level otherwise)
static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}
static void install_quiet_log(void) { llama_log_set(quiet_log, NULL); }
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// LLM owns one model + context. All calls are serialized by mu — mirrors the
// single llama worker thread in the Python engine (llama.cpp contexts are not
// thread-safe).
type LLM struct {
	mu    sync.Mutex
	model *C.struct_llama_model
	ctx   *C.struct_llama_context
	vocab *C.struct_llama_vocab

	nCtx    int
	nBatch  int
	prefix  string // system prefix currently cached in the KV
	nPrefix int    // token count of the cached prefix
	nPast   int    // tokens currently in the KV (position of next token)

	stopIDs map[C.llama_token]bool // first tokens of "<|end|>", "<|user|>"
}

type Config struct {
	ModelPath  string
	NCtx       int
	NBatch     int
	NThreads   int
	NGpuLayers int
}

// Result of one two-pass turn.
type Result struct {
	JSON  string        // pass-1 grammar-constrained JSON text
	Voice string        // pass-2 spoken acknowledgment
	TJSON time.Duration // pass-1 generation time
	TTTS  time.Duration // pass-2 generation time
	TEval time.Duration // tail eval time (prefix reuse path)
}

func Load(cfg Config) (*LLM, error) {
	C.install_quiet_log()
	C.llama_backend_init()

	cPath := C.CString(cfg.ModelPath)
	defer C.free(unsafe.Pointer(cPath))

	mp := C.llama_model_default_params()
	mp.n_gpu_layers = C.int32_t(cfg.NGpuLayers)
	model := C.llama_model_load_from_file(cPath, mp)
	if model == nil {
		return nil, fmt.Errorf("llm: cannot load model %s", cfg.ModelPath)
	}

	cp := C.llama_context_default_params()
	cp.n_ctx = C.uint32_t(cfg.NCtx)
	cp.n_batch = C.uint32_t(cfg.NBatch)
	cp.n_threads = C.int32_t(cfg.NThreads)
	cp.n_threads_batch = C.int32_t(cfg.NThreads)
	ctx := C.llama_init_from_model(model, cp)
	if ctx == nil {
		C.llama_model_free(model)
		return nil, errors.New("llm: cannot create context")
	}

	l := &LLM{
		model: model, ctx: ctx,
		vocab:  C.llama_model_get_vocab(model),
		nCtx:   cfg.NCtx,
		nBatch: cfg.NBatch,
	}
	l.stopIDs = l.phi3StopIDs()
	return l, nil
}

func (l *LLM) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.ctx != nil {
		C.llama_free(l.ctx)
		l.ctx = nil
	}
	if l.model != nil {
		C.llama_model_free(l.model)
		l.model = nil
	}
}

// ── tokenizer plumbing ────────────────────────────────────────────────────────

func (l *LLM) tokenize(text string, addSpecial bool) []C.llama_token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	buf := make([]C.llama_token, len(text)+16)
	n := C.llama_tokenize(l.vocab, cText, C.int32_t(len(text)),
		&buf[0], C.int32_t(len(buf)), C.bool(addSpecial), C.bool(true))
	if n < 0 {
		buf = make([]C.llama_token, -n)
		n = C.llama_tokenize(l.vocab, cText, C.int32_t(len(text)),
			&buf[0], C.int32_t(len(buf)), C.bool(addSpecial), C.bool(true))
	}
	return buf[:n]
}

func (l *LLM) detokenize(toks []C.llama_token) string {
	if len(toks) == 0 {
		return ""
	}
	buf := make([]C.char, len(toks)*16+16)
	n := C.llama_detokenize(l.vocab, &toks[0], C.int32_t(len(toks)),
		&buf[0], C.int32_t(len(buf)), C.bool(false), C.bool(false))
	if n < 0 {
		buf = make([]C.char, -n)
		n = C.llama_detokenize(l.vocab, &toks[0], C.int32_t(len(toks)),
			&buf[0], C.int32_t(len(buf)), C.bool(false), C.bool(false))
	}
	return C.GoStringN(&buf[0], n)
}

// phi3StopIDs ports two_pass._phi3_stop_ids: register only stop markers whose
// first token is a genuine single-token marker, skipping any that tokenize to
// a leading SentencePiece "▁" (which would truncate JSON after `"area":`).
func (l *LLM) phi3StopIDs() map[C.llama_token]bool {
	stops := map[C.llama_token]bool{}
	var spacePrefix C.llama_token = -1
	if probe := l.tokenize(" x", false); len(probe) > 0 {
		spacePrefix = probe[0]
	}
	for _, s := range []string{"<|end|>", "<|user|>"} {
		if toks := l.tokenize(s, false); len(toks) > 0 && toks[0] != spacePrefix {
			stops[toks[0]] = true
		}
	}
	return stops
}

// ── decode ────────────────────────────────────────────────────────────────────

// decodeTokens feeds tokens through llama_decode in n_batch chunks. Positions
// are inferred from the KV sequence (batch.pos == NULL), so this continues
// wherever the memory currently ends.
func (l *LLM) decodeTokens(toks []C.llama_token) error {
	for off := 0; off < len(toks); off += l.nBatch {
		end := off + l.nBatch
		if end > len(toks) {
			end = len(toks)
		}
		chunk := toks[off:end]
		batch := C.llama_batch_get_one(&chunk[0], C.int32_t(len(chunk)))
		if rc := C.llama_decode(l.ctx, batch); rc != 0 {
			return fmt.Errorf("llama_decode returned %d (n_past=%d)", rc, l.nPast)
		}
		l.nPast += len(chunk)
	}
	return nil
}

// ── KV prefix reuse (two_pass._prime_prefix) ─────────────────────────────────

func phi3SysPrefix(sys string) string  { return "<|system|>\n" + sys + "<|end|>\n" }
func phi3UserTail(user string) string  { return "<|user|>\n" + user + "<|end|>\n<|assistant|>\n" }

// PrimePrefix evaluates the constant system prefix ONCE and keeps it in the
// KV cache; each turn then drops only the tokens after it. Call at startup
// (pre-warm) — TwoPass re-primes automatically if the prefix changed.
func (l *LLM) PrimePrefix(systemPrompt string) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.primeLocked(systemPrompt)
}

func (l *LLM) primeLocked(systemPrompt string) (int, error) {
	if l.prefix == systemPrompt && l.nPast >= l.nPrefix && l.nPrefix > 0 {
		return l.nPrefix, nil
	}
	toks := l.tokenize(phi3SysPrefix(systemPrompt), true)
	mem := C.llama_get_memory(l.ctx)
	C.llama_memory_clear(mem, C.bool(true))
	l.nPast = 0
	if err := l.decodeTokens(toks); err != nil {
		return 0, err
	}
	l.prefix = systemPrompt
	l.nPrefix = len(toks)
	return l.nPrefix, nil
}

// ── two-pass generation (two_pass.approach_a) ────────────────────────────────

// TwoPass runs one intent turn: reuse the cached prefix, eval the user tail,
// generate grammar-constrained JSON (greedy), append the TTS cue, generate
// the spoken ack (temp 0.4 / top_k 40). The KV is CONTINUED across both
// passes — no state save/load.
func (l *LLM) TwoPass(systemPrompt, userText, gbnf string) (*Result, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	tEval0 := time.Now()
	nPrefix, err := l.primeLocked(systemPrompt)
	if err != nil {
		return nil, err
	}
	// Drop everything after the prefix, rewind, eval only the user tail.
	mem := C.llama_get_memory(l.ctx)
	C.llama_memory_seq_rm(mem, 0, C.llama_pos(nPrefix), -1)
	l.nPast = nPrefix
	if err := l.decodeTokens(l.tokenize(phi3UserTail(userText), false)); err != nil {
		return nil, err
	}
	tEval := time.Since(tEval0)

	// Pass 1: grammar-constrained JSON, greedy (production json_temp=0).
	cGbnf, cRoot := C.CString(gbnf), C.CString("root")
	defer C.free(unsafe.Pointer(cGbnf))
	defer C.free(unsafe.Pointer(cRoot))
	grammar := C.llama_sampler_init_grammar(l.vocab, cGbnf, cRoot)
	if grammar == nil {
		return nil, errors.New("llm: grammar failed to compile")
	}
	chain1 := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	C.llama_sampler_chain_add(chain1, grammar) // chain owns grammar now
	C.llama_sampler_chain_add(chain1, C.llama_sampler_init_greedy())
	defer C.llama_sampler_free(chain1)

	t0 := time.Now()
	jsonToks, err := l.generate(chain1, 80, false)
	if err != nil {
		return nil, err
	}
	tJSON := time.Since(t0)
	jsonText := strings.TrimSpace(strings.TrimSuffix(
		strings.TrimSpace(l.detokenize(jsonToks)), "<|end|>"))

	// Pass 2: TTS continuation — the JSON stays in context; append the cue and
	// keep generating.
	if err := l.decodeTokens(l.tokenize("\nSpoken response:", false)); err != nil {
		return nil, err
	}
	chain2 := C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
	C.llama_sampler_chain_add(chain2, C.llama_sampler_init_top_k(40))
	C.llama_sampler_chain_add(chain2, C.llama_sampler_init_temp(0.4))
	C.llama_sampler_chain_add(chain2, C.llama_sampler_init_dist(C.LLAMA_DEFAULT_SEED))
	defer C.llama_sampler_free(chain2)

	t0 = time.Now()
	ttsToks, err := l.generate(chain2, 30, true)
	if err != nil {
		return nil, err
	}
	tTTS := time.Since(t0)
	voice := l.detokenize(ttsToks)
	if i := strings.IndexByte(voice, '\n'); i >= 0 {
		voice = voice[:i]
	}
	voice = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(voice), "<|end|>"))

	return &Result{JSON: jsonText, Voice: voice,
		TJSON: tJSON, TTTS: tTTS, TEval: tEval}, nil
}

// generate samples tokens with the given chain until EOG / a phi-3 stop
// token / maxTokens (/ a newline in the decoded text when stopOnNewline).
func (l *LLM) generate(chain *C.struct_llama_sampler, maxTokens int,
	stopOnNewline bool) ([]C.llama_token, error) {
	var out []C.llama_token
	for len(out) < maxTokens {
		tok := C.llama_sampler_sample(chain, l.ctx, -1) // applies + accepts
		if bool(C.llama_vocab_is_eog(l.vocab, tok)) || l.stopIDs[tok] {
			break
		}
		out = append(out, tok)
		if stopOnNewline && strings.Contains(l.detokenize(out), "\n") {
			break
		}
		if rc := C.decode_one(l.ctx, tok); rc != 0 {
			return out, fmt.Errorf("llama_decode returned %d mid-generation", rc)
		}
		l.nPast++
	}
	return out, nil
}
