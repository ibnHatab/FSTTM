// Package nina — the Nina robot seams for the dog domain, Go port of
// nina_speak/fsttm_dog/backends_nina.py (which stays the Python playground
// reference).
//
// DOCTRINE (from the nina_ws contract): motion flows ONLY through /cmd_vel
// into the cmd_arbiter — the arbiter is the single motion author, and this
// engine must never grow another path to the legs. Named sport actions go
// on /api/sport/request per the H2 api-id table; navigation is a GOAL on
// /nina/nav_goal; VELOCITY intents from speech are refused outright. All
// ROS traffic leaves through the jsonl RobotLink → nina_relay, which can
// publish nothing but the contracted topics.
//
// This file: SemanticMemory over the canvas query pack — the RAG the robot
// derives from its own semantic-map discovery. Pack npz (built today on the
// dgx, later a nina_semantic scenegraph/palace factory product):
//
//	w1,b1,w2,b2  SigLIP adapter (two 1x1 convs = matmuls, ReLU≈GELU)
//	mu,V         PCA16 decode
//	feat16 (N,16), xyz (N,3)
//
// Query: decode → adapter → cosine vs text embedding → 0.995-quantile
// threshold → greedy 0.8 m clustering → ≤5 Candidates (map frame).
// Text embeddings: phrase_cache.npz today; the SigLIP text tower runs on
// the FSTTM host next to phi3 later. Load-once-at-boot is the accepted v0;
// /nina/instance_events will carry live increments when palace v1 lands.
package nina

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/intent"
	"github.com/ibnHatab/fsttm/go/internal/npz"
)

type Canvas struct {
	emb     [][]float32 // (M,768) adapter output, L2-normalized, z-filtered
	xyz     [][3]float32
	phrases map[string][]float32 // normalized text embeddings
}

// LoadCanvas reads the query pack (+ optional phrase cache).
func LoadCanvas(packPath, phrasePath string) (*Canvas, error) {
	p, err := npz.Load(packPath)
	if err != nil {
		return nil, fmt.Errorf("canvas: %w", err)
	}
	for _, k := range []string{"w1", "b1", "w2", "b2", "mu", "V", "feat16", "xyz"} {
		if p[k] == nil {
			return nil, fmt.Errorf("canvas: pack missing %s", k)
		}
	}
	feat16, V, mu := p["feat16"], p["V"], p["mu"]
	n, d16 := feat16.Rows(), feat16.Cols()
	d384 := V.Rows()

	// f384 = feat16 @ V.T + mu    (V is (384,16) as in the reference)
	f384 := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, d384)
		fr := feat16.Row(i)
		for j := 0; j < d384; j++ {
			var s float32
			vr := V.Row(j)
			for k := 0; k < d16; k++ {
				s += fr[k] * vr[k]
			}
			row[j] = s + mu.Data[j]
		}
		f384[i] = row
	}
	// h = relu(f384 @ w1.T + b1); e = h @ w2.T + b2
	w1, b1, w2, b2 := p["w1"], p["b1"], p["w2"], p["b2"]
	hidden, out := w1.Rows(), w2.Rows()
	c := &Canvas{}
	for i := 0; i < n; i++ {
		h := make([]float32, hidden)
		for j := 0; j < hidden; j++ {
			var s float32
			wr := w1.Row(j)
			for k := range f384[i] {
				s += f384[i][k] * wr[k]
			}
			if s += b1.Data[j]; s > 0 {
				h[j] = s
			}
		}
		e := make([]float32, out)
		var norm float64
		for j := 0; j < out; j++ {
			var s float32
			wr := w2.Row(j)
			for k := range h {
				s += h[k] * wr[k]
			}
			e[j] = s + b2.Data[j]
			norm += float64(e[j]) * float64(e[j])
		}
		norm = math.Sqrt(norm)
		if norm < 1e-8 {
			norm = 1e-8
		}
		for j := range e {
			e[j] = float32(float64(e[j]) / norm)
		}
		z := p["xyz"].Row(i)[2]
		if z > 0.15 && z < 1.8 { // reference z-filter
			c.emb = append(c.emb, e)
			r := p["xyz"].Row(i)
			c.xyz = append(c.xyz, [3]float32{r[0], r[1], r[2]})
		}
	}

	c.phrases = map[string][]float32{}
	if phrasePath != "" {
		ph, err := npz.Load(phrasePath)
		if err != nil {
			return nil, fmt.Errorf("phrase cache: %w", err)
		}
		for k, a := range ph {
			c.phrases[strings.ToLower(strings.TrimSpace(k))] = l2norm(a.Data)
		}
	}
	log.Printf("[canvas] loaded: %d cells, %d cached phrases",
		len(c.xyz), len(c.phrases))
	return c, nil
}

func l2norm(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	if n < 1e-8 {
		n = 1e-8
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out
}

// Query implements intent.SemanticMemory — the reference math verbatim.
func (c *Canvas) Query(target intent.SemanticTarget, _ []intent.Relation) []intent.Candidate {
	key := strings.ToLower(strings.TrimSpace(target.Description))
	tv, ok := c.phrases[key]
	if !ok {
		log.Printf("[canvas] no text embedding for %q (host text-tower "+
			"missing and phrase not cached)", key)
		return nil
	}
	sim := make([]float32, len(c.emb))
	for i, e := range c.emb {
		var s float32
		for j := range e {
			s += e[j] * tv[j]
		}
		sim[i] = s
	}
	thr := quantile(sim, 0.995)
	var idx []int
	for i, s := range sim {
		if s >= thr {
			idx = append(idx, i)
		}
	}
	// greedy radius clustering → instances (order by descending score)
	order := make([]int, len(idx))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return sim[idx[order[a]]] > sim[idx[order[b]]]
	})
	used := make([]bool, len(idx))
	var out []intent.Candidate
	for _, k := range order {
		if used[k] {
			continue
		}
		var cx, cy, cz, sc float64
		var m int
		kx, ky := c.xyz[idx[k]][0], c.xyz[idx[k]][1]
		for j := range idx {
			dx := float64(c.xyz[idx[j]][0] - kx)
			dy := float64(c.xyz[idx[j]][1] - ky)
			if math.Hypot(dx, dy) < 0.8 {
				used[j] = true
				cx += float64(c.xyz[idx[j]][0])
				cy += float64(c.xyz[idx[j]][1])
				cz += float64(c.xyz[idx[j]][2])
				sc += float64(sim[idx[j]])
				m++
			}
		}
		out = append(out, intent.Candidate{
			InstanceID: len(out),
			Score:      sc / float64(m),
			Position: [3]float64{cx / float64(m), cy / float64(m),
				cz / float64(m)},
		})
		if len(out) >= 5 {
			break
		}
	}
	log.Printf("[canvas] query %q -> %d candidates (best %.3f) at %s",
		target.Description, len(out),
		best(out), time.Now().Format("15:04:05"))
	return out
}

func best(c []intent.Candidate) float64 {
	if len(c) == 0 {
		return 0
	}
	return c[0].Score
}

// quantile matches numpy.quantile's default linear interpolation.
func quantile(v []float32, q float64) float32 {
	s := append([]float32(nil), v...)
	sort.Slice(s, func(a, b int) bool { return s[a] < s[b] })
	if len(s) == 0 {
		return 0
	}
	pos := q * float64(len(s)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return s[lo]
	}
	frac := float32(pos - float64(lo))
	return s[lo]*(1-frac) + s[hi]*frac
}
