package llm

// Context-rollback e2e (N09-1071 invariant 2): every turn rewinds the KV to
// the constant system prefix, so ANY earlier turn — finished or discarded —
// leaves no trace. Byte-identical JSON for the same utterance regardless of
// what ran before it is the proof. Gated on the model file.

import (
	"os"
	"testing"
)

const modelPath = "/home/axadmin/repo/vox/FSTTM/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

func loadForTest(t *testing.T) (*LLM, string, string) {
	t.Helper()
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not present")
	}
	prompt, err := os.ReadFile("../../grammar/dog-prompt.txt")
	if err != nil {
		t.Fatal(err)
	}
	gbnf, err := os.ReadFile("../../grammar/dog.gbnf")
	if err != nil {
		t.Fatal(err)
	}
	l, err := Load(Config{ModelPath: modelPath, NCtx: 2048, NBatch: 2048,
		NThreads: 6, NGpuLayers: 99})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(l.Close)
	return l, string(prompt), string(gbnf)
}

func TestRollbackYieldsByteIdenticalJSON(t *testing.T) {
	l, prompt, gbnf := loadForTest(t)
	if _, err := l.PrimePrefix(prompt); err != nil {
		t.Fatal(err)
	}

	turn := func(utt string) string {
		r, err := l.TwoPass(prompt, utt, gbnf)
		if err != nil {
			t.Fatal(err)
		}
		return r.JSON
	}

	// A, then unrelated turns of different lengths, then A again — pass 1 is
	// greedy, so ANY KV contamination from B/C would change A's bytes.
	a1 := turn("go to the chair next to the window")
	_ = turn("find the fire extinguisher")
	_ = turn("turn left ninety degrees and then stretch a little bit")
	a2 := turn("go to the chair next to the window")
	if a1 != a2 {
		t.Fatalf("KV rollback leaked context:\n a1=%s\n a2=%s", a1, a2)
	}

	// bookkeeping: nPast must equal the prefix after re-priming
	n1 := l.nPrefix
	if n, err := l.PrimePrefix(prompt); err != nil || n != n1 {
		t.Fatalf("re-prime: n=%d want %d (err=%v)", n, n1, err)
	}
}

func TestPrefixSwitchRepriming(t *testing.T) {
	l, prompt, gbnf := loadForTest(t)
	a1json, err := l.TwoPass(prompt, "sit down", gbnf)
	if err != nil {
		t.Fatal(err)
	}
	// a different prefix forces a full re-prime…
	altPrompt := prompt + "\nAlways be extremely brief."
	if _, err := l.TwoPass(altPrompt, "sit down", gbnf); err != nil {
		t.Fatal(err)
	}
	// …and switching BACK must reproduce the original bytes exactly
	a2json, err := l.TwoPass(prompt, "sit down", gbnf)
	if err != nil {
		t.Fatal(err)
	}
	if a1json.JSON != a2json.JSON {
		t.Fatalf("prefix switch contaminated the KV:\n %s\n %s",
			a1json.JSON, a2json.JSON)
	}
}
