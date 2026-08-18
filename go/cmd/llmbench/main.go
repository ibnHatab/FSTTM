// llmbench — Go two-pass parity/latency benchmark vs fsttm/two_pass.py.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/llm"
)

func main() {
	model := flag.String("model", "", "GGUF model path")
	promptF := flag.String("prompt", "", "system prompt file")
	gbnfF := flag.String("gbnf", "", "GBNF grammar file")
	ngl := flag.Int("ngl", 99, "n_gpu_layers")
	flag.Parse()

	prompt, err := os.ReadFile(*promptF)
	if err != nil {
		panic(err)
	}
	gbnf, err := os.ReadFile(*gbnfF)
	if err != nil {
		panic(err)
	}

	l, err := llm.Load(llm.Config{ModelPath: *model, NCtx: 2048, NBatch: 2048,
		NThreads: 6, NGpuLayers: *ngl})
	if err != nil {
		panic(err)
	}
	defer l.Close()

	t0 := time.Now()
	n, err := l.PrimePrefix(string(prompt))
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "prime: %d tok in %dms\n", n, time.Since(t0).Milliseconds())

	for _, utt := range flag.Args() {
		r, err := l.TwoPass(string(prompt), utt, string(gbnf))
		if err != nil {
			panic(err)
		}
		fmt.Printf("%-40q json=%4dms tts=%4dms eval=%3dms  %s\n",
			utt, r.TJSON.Milliseconds(), r.TTTS.Milliseconds(),
			r.TEval.Milliseconds(), r.JSON)
		fmt.Printf("%42s voice → %q\n", "", r.Voice)
	}
}
