// Package g2p is misaki's English G2P (misaki/en.py, Apache-2.0, the G2P Kokoro's and
// Paradee's training data were made with) ported to Go: text → the phoneme string the
// Paradee/Kokoro vocabulary encodes. American English (the af_/am_ voices).
//
// What differs from misaki: spaCy's tokenizer and tagger are replaced by a small rule
// tokenizer and tagger (tokenize.go) that reproduce the tags misaki's rules read (DT, PRP,
// IN, TO, NNP, CD, $, punctuation, the noun/verb/adjective guesses its heteronyms need);
// num2words is ported for English (num.go); the espeak fallback calls libespeak-ng
// (espeak.go). The parity test (g2p_test.go) runs the corpus misaki produced on Pascal.
package g2p

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

const (
	primary   = 'ˈ'
	secondary = 'ˌ'
)

var (
	diphthongs    = set("AIOQWYʤʧ")
	vowels        = set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
	consonants    = set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
	usTaus        = set("AIOWYiuæɑəɛɪɹʊʌ")
	subtokenJunks = set("',-._‘’/")
	puncts        = set(";:,.!?—…\"“”")
	nonQuotePunct = set(";:,.!?—…")
	ordinals      = map[string]bool{"st": true, "nd": true, "rd": true, "th": true}
	addSymbols    = map[string]string{".": "dot", "/": "slash"}
	symbols       = map[string]string{"%": "percent", "&": "and", "+": "plus", "@": "at"}
	currencies    = map[string][2]string{"$": {"dollar", "cent"}, "£": {"pound", "pence"}, "€": {"euro", "cent"}}
	punctTags     = map[string]bool{".": true, ",": true, "-LRB-": true, "-RRB-": true, "``": true, `""`: true, "''": true, ":": true, "$": true, "#": true, "NFP": true}
	punctTagPh    = map[string]string{"-LRB-": "(", "-RRB-": ")", "``": "“", `""`: "”", "''": "”"}
)

func set(s string) map[rune]bool {
	m := map[rune]bool{}
	for _, r := range s {
		m[r] = true
	}
	return m
}

// entry is a lexicon value: one pronunciation, or one per tag (with DEFAULT; a nil value
// is Python's None).
type entry struct {
	s   *string
	tag map[string]*string
}

func (e *entry) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		e.s = &s
		return nil
	}
	return json.Unmarshal(b, &e.tag)
}

// Lexicon is misaki's Lexicon (american).
type Lexicon struct {
	golds, silvers map[string]*entry
}

func loadDict(path string) (map[string]*entry, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d map[string]*entry
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return growDictionary(d), nil
}

// growDictionary is Lexicon.grow_dictionary: lowercase words also capitalized, capitalized
// ones also lowercase; the original entries win.
func growDictionary(d map[string]*entry) map[string]*entry {
	keys := make([]string, 0, len(d))
	for k := range d {
		keys = append(keys, k)
	}
	sort.Strings(keys) // dict order does not matter: e only adds keys d lacks, d wins
	out := make(map[string]*entry, len(d)*2)
	for _, k := range keys {
		if len([]rune(k)) < 2 {
			continue
		}
		v := d[k]
		if k == strings.ToLower(k) {
			if c := capitalize(k); c != k {
				out[c] = v
			}
		} else if k == capitalize(strings.ToLower(k)) {
			out[strings.ToLower(k)] = v
		}
	}
	for k, v := range d {
		out[k] = v
	}
	return out
}

// LoadLexicon reads us_gold.json and us_silver.json from dir.
func LoadLexicon(dir string) (*Lexicon, error) {
	g, err := loadDict(filepath.Join(dir, "us_gold.json"))
	if err != nil {
		return nil, err
	}
	s, err := loadDict(filepath.Join(dir, "us_silver.json"))
	if err != nil {
		return nil, err
	}
	return &Lexicon{golds: g, silvers: s}, nil
}

// ── Python string helpers ──────────────────────────────────────────────────────────────

func capitalize(s string) string {
	r := []rune(s)
	if len(r) == 0 {
		return s
	}
	return string(unicode.ToUpper(r[0])) + strings.ToLower(string(r[1:]))
}

func isAlpha(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func isDigitStr(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func lexiconOrd(r rune) bool {
	return r == 39 || r == 45 || (r >= 65 && r <= 90) || (r >= 97 && r <= 122)
}

func allLexiconOrds(s string) bool {
	for _, r := range s {
		if !lexiconOrd(r) {
			return false
		}
	}
	return true
}

func containsAny(s string, m map[rune]bool) bool {
	for _, r := range s {
		if m[r] {
			return true
		}
	}
	return false
}

// ── stress ─────────────────────────────────────────────────────────────────────────────

// Stress is misaki's optional stress number (nil: None).
type Stress = *float64

func st(v float64) Stress { return &v }

func restress(ps string) string {
	type item struct {
		key float64
		r   rune
	}
	rs := []rune(ps)
	items := make([]item, len(rs))
	for i, r := range rs {
		items[i] = item{float64(i), r}
	}
	for i, r := range rs {
		if r == primary || r == secondary {
			for j := i; j < len(rs); j++ {
				if vowels[rs[j]] {
					items[i].key = float64(j) - 0.5
					break
				}
			}
		}
	}
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].key != items[b].key {
			return items[a].key < items[b].key
		}
		return items[a].r < items[b].r
	})
	var b strings.Builder
	for _, it := range items {
		b.WriteRune(it.r)
	}
	return b.String()
}

func applyStress(ps string, stress Stress) string {
	if stress == nil {
		return ps
	}
	s := *stress
	hasP, hasS := strings.ContainsRune(ps, primary), strings.ContainsRune(ps, secondary)
	switch {
	case s < -1:
		return strings.ReplaceAll(strings.ReplaceAll(ps, string(primary), ""), string(secondary), "")
	case s == -1 || ((s == 0 || s == -0.5) && hasP):
		return strings.ReplaceAll(strings.ReplaceAll(ps, string(secondary), ""), string(primary), string(secondary))
	case (s == 0 || s == 0.5 || s == 1) && !hasP && !hasS:
		if !containsAny(ps, vowels) {
			return ps
		}
		return restress(string(secondary) + ps)
	case s >= 1 && !hasP && hasS:
		return strings.ReplaceAll(ps, string(secondary), string(primary))
	case s > 1 && !hasP && !hasS:
		if !containsAny(ps, vowels) {
			return ps
		}
		return restress(string(primary) + ps)
	}
	return ps
}

// ── the lexicon's rules ────────────────────────────────────────────────────────────────

// tokenContext is misaki's TokenContext: future_vowel is None / false / true.
type tokenContext struct {
	futureVowel *bool
	futureTo    bool
}

func (l *Lexicon) gold(w string) (string, bool) {
	e, ok := l.golds[w]
	if !ok || e.s == nil {
		return "", false
	}
	return *e.s, true
}

func (l *Lexicon) getNNP(word string) (string, int, bool) {
	var b strings.Builder
	for _, r := range word {
		if !unicode.IsLetter(r) {
			continue
		}
		g, ok := l.gold(strings.ToUpper(string(r)))
		if !ok {
			return "", 0, false
		}
		b.WriteString(g)
	}
	ps := applyStress(b.String(), st(0))
	if i := strings.LastIndex(ps, string(secondary)); i >= 0 {
		ps = ps[:i] + string(primary) + ps[i+len(string(secondary)):]
	}
	return ps, 3, true
}

func parentTag(tag string) string {
	switch {
	case strings.HasPrefix(tag, "VB"):
		return "VERB"
	case strings.HasPrefix(tag, "NN"):
		return "NOUN"
	case strings.HasPrefix(tag, "ADV"), strings.HasPrefix(tag, "RB"):
		return "ADV"
	case strings.HasPrefix(tag, "ADJ"), strings.HasPrefix(tag, "JJ"):
		return "ADJ"
	}
	return tag
}

func (l *Lexicon) specialCase(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	fv := func() *bool {
		if ctx == nil {
			return nil
		}
		return ctx.futureVowel
	}
	switch {
	case tag == "ADD" && addSymbols[word] != "":
		return l.lookup(addSymbols[word], "", st(-0.5), ctx)
	case symbols[word] != "":
		return l.lookup(symbols[word], "", nil, ctx)
	case strings.Contains(strings.Trim(word, "."), ".") && isAlpha(strings.ReplaceAll(word, ".", "")) && maxPartLen(word) < 3:
		return l.getNNP(word)
	case word == "a" || word == "A":
		if tag == "DT" {
			return "ɐ", 4, true
		}
		return "ˈA", 4, true
	case word == "am" || word == "Am" || word == "AM":
		if strings.HasPrefix(tag, "NN") {
			return l.getNNP(word)
		} else if fv() == nil || word != "am" || (stress != nil && *stress > 0) {
			g, _ := l.gold("am")
			return g, 4, true
		}
		return "ɐm", 4, true
	case word == "an" || word == "An" || word == "AN":
		if word == "AN" && strings.HasPrefix(tag, "NN") {
			return l.getNNP(word)
		}
		return "ɐn", 4, true
	case word == "I" && tag == "PRP":
		return string(secondary) + "I", 4, true
	case (word == "by" || word == "By" || word == "BY") && parentTag(tag) == "ADV":
		return "bˈI", 4, true
	case word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")):
		v := fv()
		if v == nil {
			g, _ := l.gold("to")
			return g, 4, true
		} else if !*v {
			return "tə", 4, true
		}
		return "tʊ", 4, true
	case word == "in" || word == "In" || (word == "IN" && tag != "NNP"):
		s := ""
		if fv() == nil || tag != "IN" {
			s = string(primary)
		}
		return s + "ɪn", 4, true
	case word == "the" || word == "The" || (word == "THE" && tag == "DT"):
		if v := fv(); v != nil && *v {
			return "ði", 4, true
		}
		return "ðə", 4, true
	case tag == "IN" && (strings.EqualFold(word, "vs") || strings.EqualFold(word, "vs.")):
		return l.lookup("versus", "", nil, ctx)
	case word == "used" || word == "Used" || word == "USED":
		e := l.golds["used"]
		if (tag == "VBD" || tag == "JJ") && ctx != nil && ctx.futureTo {
			return *e.tag["VBD"], 4, true
		}
		return *e.tag["DEFAULT"], 4, true
	}
	return "", 0, false
}

func maxPartLen(word string) int {
	m := 0
	for _, p := range strings.Split(word, ".") {
		if n := len([]rune(p)); n > m {
			m = n
		}
	}
	return m
}

func (l *Lexicon) isKnown(word, tag string) bool {
	if _, ok := l.golds[word]; ok {
		return true
	}
	if symbols[word] != "" {
		return true
	}
	if _, ok := l.silvers[word]; ok {
		return true
	}
	if !isAlpha(word) || !allLexiconOrds(word) {
		return false
	}
	if len([]rune(word)) == 1 {
		return true
	}
	if word == strings.ToUpper(word) {
		if _, ok := l.golds[strings.ToLower(word)]; ok {
			return true
		}
	}
	r := []rune(word)
	return string(r[1:]) == strings.ToUpper(string(r[1:]))
}

func (l *Lexicon) lookup(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	isNNP := false
	if word == strings.ToUpper(word) {
		if _, ok := l.golds[word]; !ok {
			word = strings.ToLower(word)
			isNNP = tag == "NNP"
		}
	}
	var ps *string
	rating := 4
	e, ok := l.golds[word]
	if !ok && !isNNP {
		e, ok = l.silvers[word]
		rating = 3
	}
	if ok {
		if e.s != nil {
			ps = e.s
		} else if e.tag != nil {
			t := tag
			if _, has := e.tag["None"]; ctx != nil && ctx.futureVowel == nil && has {
				t = "None"
			} else if _, has := e.tag[t]; !has {
				t = parentTag(t)
			}
			if v, has := e.tag[t]; has {
				ps = v
			} else {
				ps = e.tag["DEFAULT"]
			}
		}
	}
	if ps == nil || (isNNP && !strings.ContainsRune(*ps, primary)) {
		if p, r, ok := l.getNNP(word); ok {
			return p, r, true
		}
	}
	if ps == nil {
		return "", 0, false
	}
	return applyStress(*ps, stress), rating, true
}

func (l *Lexicon) suffS(stem string) string {
	if stem == "" {
		return ""
	}
	r := []rune(stem)
	last := r[len(r)-1]
	switch {
	case strings.ContainsRune("ptkfθ", last):
		return stem + "s"
	case strings.ContainsRune("szʃʒʧʤ", last):
		return stem + "ᵻz"
	}
	return stem + "z"
}

func (l *Lexicon) stemS(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	n := len([]rune(word))
	if n < 3 || !strings.HasSuffix(word, "s") {
		return "", 0, false
	}
	var stem string
	switch {
	case !strings.HasSuffix(word, "ss") && l.isKnown(word[:len(word)-1], tag):
		stem = word[:len(word)-1]
	case (strings.HasSuffix(word, "'s") || (n > 4 && strings.HasSuffix(word, "es") && !strings.HasSuffix(word, "ies"))) && l.isKnown(word[:len(word)-2], tag):
		stem = word[:len(word)-2]
	case n > 4 && strings.HasSuffix(word, "ies") && l.isKnown(word[:len(word)-3]+"y", tag):
		stem = word[:len(word)-3] + "y"
	default:
		return "", 0, false
	}
	ps, r, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", r, false
	}
	return l.suffS(ps), r, true
}

func (l *Lexicon) suffEd(stem string) string {
	if stem == "" {
		return ""
	}
	r := []rune(stem)
	last := r[len(r)-1]
	switch {
	case strings.ContainsRune("pkfθʃsʧ", last):
		return stem + "t"
	case last == 'd':
		return stem + "ᵻd"
	case last != 't':
		return stem + "d"
	case len(r) < 2:
		return stem + "ɪd"
	case usTaus[r[len(r)-2]]:
		return string(r[:len(r)-1]) + "ɾᵻd"
	}
	return stem + "ᵻd"
}

func (l *Lexicon) stemEd(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	n := len([]rune(word))
	if n < 4 || !strings.HasSuffix(word, "d") {
		return "", 0, false
	}
	var stem string
	switch {
	case !strings.HasSuffix(word, "dd") && l.isKnown(word[:len(word)-1], tag):
		stem = word[:len(word)-1]
	case n > 4 && strings.HasSuffix(word, "ed") && !strings.HasSuffix(word, "eed") && l.isKnown(word[:len(word)-2], tag):
		stem = word[:len(word)-2]
	default:
		return "", 0, false
	}
	ps, r, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", r, false
	}
	return l.suffEd(ps), r, true
}

func (l *Lexicon) suffIng(stem string) string {
	if stem == "" {
		return ""
	}
	r := []rune(stem)
	if len(r) > 1 && r[len(r)-1] == 't' && usTaus[r[len(r)-2]] {
		return string(r[:len(r)-1]) + "ɾɪŋ"
	}
	return stem + "ɪŋ"
}

func doubledIng(word string) bool {
	if strings.HasSuffix(word, "cking") {
		return true
	}
	r := []rune(word)
	if len(r) < 5 || !strings.HasSuffix(word, "ing") {
		return false
	}
	a, b := r[len(r)-5], r[len(r)-4]
	return a == b && strings.ContainsRune("bcdgklmnprstvxz", a)
}

func (l *Lexicon) stemIng(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	n := len([]rune(word))
	if n < 5 || !strings.HasSuffix(word, "ing") {
		return "", 0, false
	}
	base := word[:len(word)-3]
	var stem string
	switch {
	case n > 5 && l.isKnown(base, tag):
		stem = base
	case l.isKnown(base+"e", tag):
		stem = base + "e"
	case n > 5 && doubledIng(word) && l.isKnown(word[:len(word)-4], tag):
		stem = word[:len(word)-4]
	default:
		return "", 0, false
	}
	ps, r, ok := l.lookup(stem, tag, stress, ctx)
	if !ok {
		return "", r, false
	}
	return l.suffIng(ps), r, true
}

func (l *Lexicon) getWord(word, tag string, stress Stress, ctx *tokenContext) (string, int, bool) {
	if ps, r, ok := l.specialCase(word, tag, stress, ctx); ok {
		return ps, r, true
	}
	wl := strings.ToLower(word)
	if len([]rune(word)) > 1 && isAlpha(strings.ReplaceAll(word, "'", "")) && word != wl &&
		(tag != "NNP" || len([]rune(word)) > 7) && l.golds[word] == nil && l.silvers[word] == nil {
		r := []rune(word)
		if word == strings.ToUpper(word) || string(r[1:]) == strings.ToLower(string(r[1:])) {
			_, g := l.golds[wl]
			_, s := l.silvers[wl]
			if g || s || l.anyStem(wl, tag, stress, ctx) {
				word = wl
			}
		}
	}
	if l.isKnown(word, tag) {
		return l.lookup(word, tag, stress, ctx)
	}
	if strings.HasSuffix(word, "s'") && l.isKnown(word[:len(word)-2]+"'s", tag) {
		return l.lookup(word[:len(word)-2]+"'s", tag, stress, ctx)
	}
	if strings.HasSuffix(word, "'") && l.isKnown(word[:len(word)-1], tag) {
		return l.lookup(word[:len(word)-1], tag, stress, ctx)
	}
	if ps, r, ok := l.stemS(word, tag, stress, ctx); ok {
		return ps, r, true
	}
	if ps, r, ok := l.stemEd(word, tag, stress, ctx); ok {
		return ps, r, true
	}
	s := stress
	if s == nil {
		s = st(0.5)
	}
	if ps, r, ok := l.stemIng(word, tag, s, ctx); ok {
		return ps, r, true
	}
	return "", 0, false
}

func (l *Lexicon) anyStem(wl, tag string, stress Stress, ctx *tokenContext) bool {
	if _, _, ok := l.stemS(wl, tag, stress, ctx); ok {
		return true
	}
	if _, _, ok := l.stemEd(wl, tag, stress, ctx); ok {
		return true
	}
	_, _, ok := l.stemIng(wl, tag, stress, ctx)
	return ok
}

func (l *Lexicon) appendCurrency(ps, currency string) string {
	if currency == "" {
		return ps
	}
	c, ok := currencies[currency]
	if !ok {
		return ps
	}
	if s, _, ok := l.stemS(c[0]+"s", "", nil, nil); ok {
		return ps + " " + s
	}
	return ps
}

func isNumberWord(word string, isHead bool) bool {
	hasDigit := false
	for _, r := range word {
		if r >= '0' && r <= '9' {
			hasDigit = true
		}
	}
	if !hasDigit {
		return false
	}
	for _, s := range []string{"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"} {
		if strings.HasSuffix(word, s) {
			word = word[:len(word)-len(s)]
			break
		}
	}
	for i, r := range word {
		if !((r >= '0' && r <= '9') || r == ',' || r == '.' || (isHead && i == 0 && r == '-')) {
			return false
		}
	}
	return true
}

// word is the lexicon applied to one token (Lexicon.__call__).
func (l *Lexicon) word(tk *mtoken, ctx *tokenContext) (string, int, bool) {
	w := tk.text
	if tk.alias != nil {
		w = *tk.alias
	}
	w = strings.NewReplacer("‘", "'", "’", "'").Replace(w)
	var stress Stress
	if w != strings.ToLower(w) {
		if w == strings.ToUpper(w) {
			stress = st(2)
		} else {
			stress = st(0.5)
		}
	}
	if ps, r, ok := l.getWord(w, tk.tag, stress, ctx); ok {
		return applyStress(l.appendCurrency(ps, tk.currency), tk.stress), r, true
	}
	if isNumberWord(w, tk.isHead) {
		ps, r, ok := l.getNumber(w, tk.currency, tk.isHead, tk.numFlags)
		if !ok {
			return "", 0, false
		}
		return applyStress(ps, tk.stress), r, true
	}
	return "", 0, false
}
