package g2p

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The parity corpus: sentences typical of Nina (the home rules' sentences, her logged
// replies, numbers, times, money, heteronyms), each with misaki's phonemes and spaCy's
// tokens and tags, made on Pascal by the reference misaki 0.9.4 (paradee's venv) and
// espeak-ng 1.52. The lexicons are not in git: $G2P_LEXICON, else models/paradee/.
type sample struct {
	Text  string     `json:"text"`
	Ps    string     `json:"ps"`
	Spacy [][]string `json:"spacy"`
}

func lexDir(t *testing.T) string {
	d := os.Getenv("G2P_LEXICON")
	if d == "" {
		d = filepath.Join("..", "..", "..", "models", "paradee")
	}
	if _, err := os.Stat(filepath.Join(d, "us_gold.json")); err != nil {
		t.Skipf("no lexicon at %s (set G2P_LEXICON)", d)
	}
	return d
}

func corpus(t *testing.T) []sample {
	b, err := os.ReadFile("testdata/misaki_corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	var s []sample
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestParity(t *testing.T) {
	es, err := NewEspeak()
	if err != nil {
		t.Skip(err)
	}
	g, err := New(lexDir(t), es.Phonemes)
	if err != nil {
		t.Fatal(err)
	}
	ok, n := 0, 0
	var diffs []string
	for _, s := range corpus(t) {
		n++
		got := g.Phonemize(s.Text)
		if got == s.Ps {
			ok++
			continue
		}
		diffs = append(diffs, s.Text+"\n   misaki: "+s.Ps+"\n   go:     "+got)
	}
	t.Logf("parity %d/%d (%.1f%%)", ok, n, 100*float64(ok)/float64(n))
	if os.Getenv("G2P_DIFFS") != "" {
		t.Log("\n" + strings.Join(diffs, "\n"))
	}
	if float64(ok)/float64(n) < 0.95 {
		t.Errorf("parity %d/%d below 95%%", ok, n)
	}
}

// TestTokenizer: the tokens against spaCy's (text only; the tags are guesses).
func TestTokenizer(t *testing.T) {
	ok, n := 0, 0
	for _, s := range corpus(t) {
		n++
		got := tokenizeText(s.Text)
		var a, b []string
		for _, x := range got {
			a = append(a, x.text)
		}
		for _, x := range s.Spacy {
			b = append(b, x[0])
		}
		if strings.Join(a, "|") == strings.Join(b, "|") {
			ok++
		} else if os.Getenv("G2P_DIFFS") != "" {
			t.Logf("%q\n   spacy: %v\n   go:    %v", s.Text, b, a)
		}
	}
	t.Logf("tokens %d/%d", ok, n)
}

func TestNumbers(t *testing.T) {
	for n, want := range map[int64]string{105: "one hundred and five", 1100: "one thousand, one hundred", 2026: "two thousand and twenty-six",
		1234567: "one million, two hundred and thirty-four thousand, five hundred and sixty-seven"} {
		if got := cardinal(n); got != want {
			t.Errorf("cardinal %d: %q want %q", n, got, want)
		}
	}
	for n, want := range map[int64]string{2026: "twenty twenty-six", 1905: "nineteen oh-five", 1900: "nineteen hundred", 2005: "two thousand and five", 2010: "twenty ten", 105: "one oh-five"} {
		if got := year(n); got != want {
			t.Errorf("year %d: %q want %q", n, got, want)
		}
	}
	for n, want := range map[int64]string{22: "twenty-second", 1: "first", 40: "fortieth", 112: "one hundred and twelfth"} {
		if got := ordinal(n); got != want {
			t.Errorf("ordinal %d: %q want %q", n, got, want)
		}
	}
	if got := floatWords("25.99"); got != "twenty-five point nine nine" {
		t.Error(got)
	}
}
