package stt

import (
	"bytes"
	"compress/zlib"
	"regexp"
	"strings"
)

// A decoder loop: the same phrase over and over ("and without the stereo channel, and without
// the stereo channel, …", "her her her …", "I think it's a great story." × 40). Two tests, either
// is a loop:
//   - a run: a phrase of n words (1..12) repeated back to back, at least minRun(n) times;
//   - the compression ratio (whisper's measure: the text's bytes over its zlib bytes) above 4, for
//     a text of 80 bytes or more. Whisper's own limit, 2.4, is for one 30 s window and flags a
//     sentence said three times (3.8): here it is the backstop, the runs catch the loops.

const compressionLimit = 4.0

var wordRe = regexp.MustCompile(`[\p{L}\p{N}']+`)

func words(text string) []string {
	w := wordRe.FindAllString(strings.ToLower(text), -1)
	return w
}

// minRun is how many back-to-back copies of an n-word phrase make a loop: a word said 5 times
// in a row is not speech ("no no no" is), a phrase said 4 times (a sentence said 3 times can be:
// a person repeating it, a song's chorus).
func minRun(n int) int {
	if n == 1 {
		return 5
	}
	return 4
}

// run finds the longest back-to-back repetition: the start word, the phrase length n and the copies.
func run(w []string) (at, n, copies int) {
	for size := 1; size <= 12; size++ {
		for i := 0; i+2*size <= len(w); i++ {
			c := 1
			for j := i + size; j+size <= len(w) && equal(w[i:i+size], w[j:j+size]); j += size {
				c++
			}
			if c >= minRun(size) && c*size > copies*n {
				at, n, copies = i, size, c
			}
		}
	}
	return
}

func equal(a, b []string) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// CompressionRatio is whisper's: the text's bytes over its zlib-compressed bytes.
func CompressionRatio(text string) float64 {
	if text == "" {
		return 0
	}
	var b bytes.Buffer
	z := zlib.NewWriter(&b)
	_, _ = z.Write([]byte(text))
	_ = z.Close()
	return float64(len(text)) / float64(b.Len())
}

// Looped says whether a transcript is a decoder loop.
func Looped(text string) bool {
	if _, n, _ := run(words(text)); n > 0 {
		return true
	}
	return len(text) >= 80 && CompressionRatio(text) > compressionLimit
}

// Collapse keeps one copy of every looped run, marked by an ellipsis: "a b c, a b c, a b c, d" →
// "a b c … d". Words outside the runs keep the transcript's spelling of them.
func Collapse(text string) string {
	toks := wordRe.FindAllStringIndex(text, -1)
	w := words(text)
	for guard := 0; guard < 20; guard++ {
		at, n, copies := run(w)
		if n == 0 {
			break
		}
		keepEnd := toks[at+n-1][1]
		var restStart int
		if at+n*copies < len(toks) {
			restStart = toks[at+n*copies][0]
		} else {
			restStart = len(text)
		}
		text = text[:keepEnd] + " … " + strings.TrimLeft(text[restStart:], " ")
		toks = wordRe.FindAllStringIndex(text, -1)
		w = words(text)
	}
	return strings.TrimSpace(text)
}
