//go:build !noparadee

// Package paradee is the Paradee TTS (sahilmahendrakar/Paradee-8M, Apache-2.0: a small
// Kokoro-style model) in Go: text → misaki phonemes (internal/g2p) → the model's vocabulary
// ids → one ONNX graph (onnxruntime's C API) → 24 kHz float audio. The reference is
// paradee/tts.py 1.2.0.
package paradee

/*
#cgo pkg-config: libonnxruntime
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>

static const OrtApi *api(void) { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

static char *errmsg(OrtStatus *s) {
	if (!s) return NULL;
	const char *m = api()->GetErrorMessage(s);
	char *c = strdup(m ? m : "onnxruntime error");
	api()->ReleaseStatus(s);
	return c;
}

typedef struct { OrtEnv *env; OrtSession *sess; OrtMemoryInfo *mem; } pd_model;

static char *pd_open(const char *path, int threads, pd_model *m) {
	char *e;
	OrtSessionOptions *so = NULL;
	if ((e = errmsg(api()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "paradee", &m->env)))) return e;
	if ((e = errmsg(api()->CreateSessionOptions(&so)))) return e;
	if ((e = errmsg(api()->SetIntraOpNumThreads(so, threads)))) return e;
	if ((e = errmsg(api()->SetInterOpNumThreads(so, 1)))) return e;
	e = errmsg(api()->CreateSession(m->env, path, so, &m->sess));
	api()->ReleaseSessionOptions(so);
	if (e) return e;
	return errmsg(api()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem));
}

static void pd_close(pd_model *m) {
	if (m->mem) api()->ReleaseMemoryInfo(m->mem);
	if (m->sess) api()->ReleaseSession(m->sess);
	if (m->env) api()->ReleaseEnv(m->env);
}

// pd_run: ids → waveform. *out is malloc'd (free with free()), *n its sample count.
static char *pd_run(pd_model *m, int64_t *ids, int64_t nids, float speed, int64_t voice, int with_voice, float **out, int64_t *n) {
	char *e = NULL;
	OrtValue *in[3] = {0}, *res = NULL;
	int64_t shape_ids[2] = {1, nids}, shape1[1] = {1};
	const char *names[3] = {"input_ids", "speed", "voice"};
	const char *outnames[1] = {"waveform"};
	if ((e = errmsg(api()->CreateTensorWithDataAsOrtValue(m->mem, ids, nids * sizeof(int64_t), shape_ids, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[0])))) goto done;
	if ((e = errmsg(api()->CreateTensorWithDataAsOrtValue(m->mem, &speed, sizeof(float), shape1, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1])))) goto done;
	if (with_voice && (e = errmsg(api()->CreateTensorWithDataAsOrtValue(m->mem, &voice, sizeof(int64_t), shape1, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[2])))) goto done;
	if ((e = errmsg(api()->Run(m->sess, NULL, names, (const OrtValue *const *)in, with_voice ? 3 : 2, outnames, 1, &res)))) goto done;
	{
		OrtTensorTypeAndShapeInfo *info = NULL;
		size_t cnt = 0;
		float *p = NULL;
		if ((e = errmsg(api()->GetTensorTypeAndShape(res, &info)))) goto done;
		e = errmsg(api()->GetTensorShapeElementCount(info, &cnt));
		api()->ReleaseTensorTypeAndShapeInfo(info);
		if (e) goto done;
		if ((e = errmsg(api()->GetTensorMutableData(res, (void **)&p)))) goto done;
		*out = (float *)malloc(cnt * sizeof(float));
		memcpy(*out, p, cnt * sizeof(float));
		*n = (int64_t)cnt;
	}
done:
	for (int i = 0; i < 3; i++) if (in[i]) api()->ReleaseValue(in[i]);
	if (res) api()->ReleaseValue(res);
	return e;
}
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unsafe"

	"github.com/ibnHatab/fsttm/go/internal/g2p"
)

const (
	SampleRate  = 24000
	maxPhonemes = 510 // the text side has 512 positions, two of them the pads
)

// The voices of 1.2, in the order the model numbers them.
var Voices = []string{"af_heart", "bf_isabella", "am_puck", "bm_fable"}

// Model is a loaded Paradee: the ONNX session, its vocabulary, the G2P.
type Model struct {
	mu    sync.Mutex
	m     C.pd_model
	vocab map[rune]int64
	g2p   *g2p.G2P
	multi bool
}

// Open loads dir/paradee_int8.onnx and dir/config.json, and the G2P's lexicons from dir.
func Open(dir string, threads int) (*Model, error) {
	cfgB, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Vocab  map[string]int64 `json:"vocab"`
		Voices []string         `json:"voices"`
	}
	if err := json.Unmarshal(cfgB, &cfg); err != nil {
		return nil, fmt.Errorf("paradee config: %w", err)
	}
	vocab := map[rune]int64{}
	for k, v := range cfg.Vocab {
		r := []rune(k)
		if len(r) == 1 {
			vocab[r[0]] = v
		}
	}
	es, err := g2p.NewEspeak()
	if err != nil {
		return nil, err
	}
	g, err := g2p.New(dir, es.Phonemes)
	if err != nil {
		return nil, err
	}
	if threads <= 0 {
		threads = 1
	}
	md := &Model{vocab: vocab, g2p: g, multi: len(cfg.Voices) > 1}
	cp := C.CString(filepath.Join(dir, "paradee_int8.onnx"))
	defer C.free(unsafe.Pointer(cp))
	if e := C.pd_open(cp, C.int(threads), &md.m); e != nil {
		defer C.free(unsafe.Pointer(e))
		C.pd_close(&md.m)
		return nil, fmt.Errorf("paradee: %s", C.GoString(e))
	}
	return md, nil
}

func (md *Model) Close() { C.pd_close(&md.m) }

// Phonemize is the G2P (American; the af_/am_ voices).
func (md *Model) Phonemize(text string) string { return md.g2p.Phonemize(text) }

// FromPhonemes is generate_from_phonemes: one chunk of phonemes → audio.
func (md *Model) FromPhonemes(ps string, speed float32, voice string) ([]float32, error) {
	vi := int64(-1)
	for i, v := range Voices {
		if v == voice {
			vi = int64(i)
		}
	}
	if vi < 0 {
		return nil, fmt.Errorf("paradee has no voice %q (%s)", voice, strings.Join(Voices, ", "))
	}
	ids := []int64{0}
	for _, c := range ps {
		if id, ok := md.vocab[c]; ok && len(ids) <= maxPhonemes {
			ids = append(ids, id)
		}
	}
	ids = append(ids, 0)
	md.mu.Lock()
	defer md.mu.Unlock()
	var out *C.float
	var n C.int64_t
	wv := 0
	if md.multi {
		wv = 1
	}
	if e := C.pd_run(&md.m, (*C.int64_t)(unsafe.Pointer(&ids[0])), C.int64_t(len(ids)), C.float(speed), C.int64_t(vi), C.int(wv), &out, &n); e != nil {
		defer C.free(unsafe.Pointer(e))
		return nil, fmt.Errorf("paradee: %s", C.GoString(e))
	}
	defer C.free(unsafe.Pointer(out))
	res := make([]float32, int(n))
	copy(res, unsafe.Slice((*float32)(unsafe.Pointer(out)), int(n)))
	return res, nil
}

var sentenceSplit = regexp.MustCompile(`([.!?…])\s+|\n+`)

// Chunks is _chunks: the text's sentences as phoneme strings, a long one cut at a space.
func (md *Model) Chunks(text string) []string {
	var out []string
	for _, s := range splitSentences(strings.TrimSpace(text)) {
		if strings.TrimSpace(s) == "" {
			continue
		}
		ps := []rune(md.Phonemize(s))
		for len(ps) > maxPhonemes {
			cut := lastSpace(ps[:maxPhonemes])
			if cut <= 0 {
				cut = maxPhonemes
			}
			out = append(out, string(ps[:cut]))
			ps = []rune(strings.TrimLeft(string(ps[cut:]), " "))
		}
		if len(ps) > 0 {
			out = append(out, string(ps))
		}
	}
	return out
}

func lastSpace(r []rune) int {
	for i := len(r) - 1; i >= 0; i-- {
		if r[i] == ' ' {
			return i
		}
	}
	return -1
}

// splitSentences is re.split(r"(?<=[.!?…])\s+|\n+", text): after sentence punctuation the
// whitespace goes, the punctuation stays.
func splitSentences(text string) []string {
	var out []string
	last := 0
	for _, m := range sentenceSplit.FindAllStringSubmatchIndex(text, -1) {
		end := m[0]
		if m[2] >= 0 { // the punctuation group: keep it with the sentence
			end = m[3]
		}
		out = append(out, text[last:end])
		last = m[1]
	}
	return append(out, text[last:])
}

// Synthesize is Paradee.__call__: the whole text, sentence by sentence, concatenated.
func (md *Model) Synthesize(text string, speed float32, voice string) ([]float32, error) {
	var all []float32
	for _, ps := range md.Chunks(text) {
		a, err := md.FromPhonemes(ps, speed, voice)
		if err != nil {
			return nil, err
		}
		all = append(all, a...)
	}
	return all, nil
}
