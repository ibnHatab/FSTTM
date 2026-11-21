//go:build nollama

// Package llm without llama.cpp (build tag nollama): the engine with the LLM over an API
// (cloud:) links no libllama, so whisper.cpp's ggml is the only ggml in the process (two
// ggml builds in one process resolve each other's symbols). Load fails; Result stays.
package llm

import (
	"errors"
	"time"
)

type Config struct {
	ModelPath  string
	NCtx       int
	NBatch     int
	NThreads   int
	NGpuLayers int
}

type Result struct {
	JSON  string
	Voice string
	TJSON time.Duration
	TTTS  time.Duration
	TEval time.Duration
}

type LLM struct{}

var errNo = errors.New("llm: built with -tags nollama (no llama.cpp): set cloud: in the config")

func Load(Config) (*LLM, error)                                   { return nil, errNo }
func (l *LLM) Close()                                             {}
func (l *LLM) PrimePrefix(string) (int, error)                    { return 0, errNo }
func (l *LLM) TwoPass(string, string, string) (*Result, error)    { return nil, errNo }
