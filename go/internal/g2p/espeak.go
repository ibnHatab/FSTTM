package g2p

/*
#cgo pkg-config: espeak-ng
#include <stdlib.h>
#include <espeak-ng/speak_lib.h>

static int g2p_espeak_init(const char *voice) {
	if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, NULL, espeakINITIALIZE_DONT_EXIT) <= 0) return -1;
	return espeak_SetVoiceByName(voice) == EE_OK ? 0 : -2;
}

// one clause's phonemes; *done set when the text is used up
static const char *g2p_espeak_clause(const void **ptr, int mode, int *done) {
	const char *ps = espeak_TextToPhonemes(ptr, espeakCHARS_UTF8, mode);
	*done = (*ptr == NULL);
	return ps;
}
*/
import "C"

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
	"unsafe"
)

// Espeak is misaki's EspeakFallback (American): the word through libespeak-ng in IPA with
// '^' ties (phonemizer's EspeakBackend(preserve_punctuation=True, with_stress=True,
// tie='^')), then misaki's espeak→misaki phoneme map.
type Espeak struct {
	mu   sync.Mutex
	mode C.int
}

// NewEspeak initializes libespeak-ng with the en-us voice (once per process).
func NewEspeak() (*Espeak, error) {
	v := C.CString("en-us")
	defer C.free(unsafe.Pointer(v))
	if rc := C.g2p_espeak_init(v); rc != 0 {
		return nil, fmt.Errorf("espeak-ng: init failed (%d)", int(rc))
	}
	// bit 1: IPA; bit 7: a tie (bits 8-23) inside multi-letter phonemes
	return &Espeak{mode: C.int(0x02 | 0x80 | int('^')<<8)}, nil
}

// phonemizer's text_to_phonemes + _postprocess_line for one chunk without punctuation.
func (e *Espeak) chunk(text string) string {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))
	ptr := unsafe.Pointer(ct)
	var parts []string
	for {
		var done C.int
		ps := C.g2p_espeak_clause(&ptr, e.mode, &done)
		if ps != nil {
			if s := C.GoString(ps); s != "" {
				parts = append(parts, s)
			}
		}
		if done != 0 || ptr == nil {
			break
		}
	}
	line := strings.TrimSpace(strings.Join(parts, " "))
	line = strings.ReplaceAll(strings.ReplaceAll(line, "\n", " "), "  ", " ")
	var words []string
	for _, w := range strings.Split(line, " ") {
		if w = strings.TrimSpace(w); w != "" {
			words = append(words, w)
		}
	}
	return strings.Join(words, " ")
}

var (
	marksRe = regexp.MustCompile(`(\s*[;:,.!?¡¿—…"«»“”(){}\[\]]+\s*)+`)
	e2m     = [][2]string{ // misaki EspeakFallback.E2M, longest first (ties: its dict order)
		{"ʔˌn̩", "ʔn"}, {"ʔn̩", "ʔn"}, {"a^ɪ", "I"}, {"a^ʊ", "W"}, {"d^ʒ", "ʤ"}, {"e^ɪ", "A"}, {"t^ʃ", "ʧ"},
		{"ɔ^ɪ", "Y"}, {"ə^l", "ᵊl"}, {"ʲo", "jo"}, {"ʲə", "jə"}, {"ʲ", ""}, {"e", "A"}, {"ɚ", "əɹ"}, {"r", "ɹ"},
		{"x", "k"}, {"ç", "k"}, {"ɐ", "ə"}, {"ɬ", "l"}, {"̃", ""},
	}
	syllabicRe = regexp.MustCompile(`(\S)\x{0329}`)
)

// Phonemes is EspeakFallback.__call__ on one token's text.
func (e *Espeak) Phonemes(text string) (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	// phonemizer's Punctuation.preserve / restore: the chunks between marks phonemized, the
	// marks put back where they were
	var b strings.Builder
	last := 0
	for _, m := range marksRe.FindAllStringIndex(text, -1) {
		if m[0] > last {
			b.WriteString(e.chunk(text[last:m[0]]))
		}
		b.WriteString(text[m[0]:m[1]])
		last = m[1]
	}
	if last < len(text) {
		b.WriteString(e.chunk(text[last:]))
	}
	ps := strings.TrimSpace(b.String())
	if ps == "" {
		return "", false
	}
	for _, kv := range e2m {
		ps = strings.ReplaceAll(ps, kv[0], kv[1])
	}
	ps = syllabicRe.ReplaceAllString(ps, "ᵊ$1")
	ps = strings.ReplaceAll(ps, "̩", "")
	ps = strings.ReplaceAll(ps, "o^ʊ", "O")
	ps = strings.ReplaceAll(ps, "ɜːɹ", "ɜɹ")
	ps = strings.ReplaceAll(ps, "ɜː", "ɜɹ")
	ps = strings.ReplaceAll(ps, "ɪə", "iə")
	ps = strings.ReplaceAll(ps, "ː", "")
	ps = strings.ReplaceAll(ps, "o", "ɔ")
	ps = strings.ReplaceAll(ps, "ɾ", "T")
	ps = strings.ReplaceAll(ps, "ʔ", "t")
	return strings.ReplaceAll(ps, "^", ""), true
}
