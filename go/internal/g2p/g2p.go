package g2p

import (
	"sort"
	"strings"
	"unicode"
)

// mtoken is misaki's MToken with its underscore fields.
type mtoken struct {
	text, tag, ws string
	phonemes      *string
	rating        int
	isHead        bool
	alias         *string
	stress        Stress
	currency      string
	numFlags      string
	prespace      bool
}

func sp(s string) *string { return &s }

// Fallback phonemizes a word the lexicon does not know (misaki's EspeakFallback).
type Fallback func(text string) (string, bool)

// G2P is misaki's en.G2P (American, no transformer).
type G2P struct {
	Lex      *Lexicon
	Fallback Fallback
	Unk      string
}

// New loads the lexicons from dir (us_gold.json, us_silver.json).
func New(dir string, fb Fallback) (*G2P, error) {
	l, err := LoadLexicon(dir)
	if err != nil {
		return nil, err
	}
	return &G2P{Lex: l, Fallback: fb}, nil
}

func mergeTokens(tks []*mtoken, unk *string) *mtoken {
	stresses := map[float64]bool{}
	cur := ""
	var flags []rune
	ratingNone := false
	rating := 99
	for _, tk := range tks {
		if tk.stress != nil {
			stresses[*tk.stress] = true
		}
		if tk.currency != "" && tk.currency > cur {
			cur = tk.currency
		}
		for _, c := range tk.numFlags {
			if !strings.ContainsRune(string(flags), c) {
				flags = append(flags, c)
			}
		}
		if tk.rating == 0 {
			ratingNone = true
		} else if tk.rating < rating {
			rating = tk.rating
		}
	}
	sort.Slice(flags, func(a, b int) bool { return flags[a] < flags[b] })
	var ph *string
	if unk != nil {
		var b strings.Builder
		for _, tk := range tks {
			s := b.String()
			if tk.prespace && s != "" && !unicode.IsSpace([]rune(s)[len([]rune(s))-1]) && tk.phonemes != nil && *tk.phonemes != "" {
				b.WriteString(" ")
			}
			if tk.phonemes == nil {
				b.WriteString(*unk)
			} else {
				b.WriteString(*tk.phonemes)
			}
		}
		ph = sp(b.String())
	}
	var text strings.Builder
	for i, tk := range tks {
		text.WriteString(tk.text)
		if i < len(tks)-1 {
			text.WriteString(tk.ws)
		}
	}
	best, bestScore := tks[0], -1
	for _, tk := range tks {
		sc := 0
		for _, c := range tk.text {
			if string(c) == strings.ToLower(string(c)) {
				sc++
			} else {
				sc += 2
			}
		}
		if sc > bestScore {
			best, bestScore = tk, sc
		}
	}
	m := &mtoken{text: text.String(), tag: best.tag, ws: tks[len(tks)-1].ws, phonemes: ph,
		isHead: tks[0].isHead, currency: cur, numFlags: string(flags), prespace: tks[0].prespace}
	if len(stresses) == 1 {
		for s := range stresses {
			m.stress = st(s)
		}
	}
	if !ratingNone {
		m.rating = rating
	}
	return m
}

func stressWeight(ps string) int {
	n := 0
	for _, c := range ps {
		if diphthongs[c] {
			n += 2
		} else {
			n++
		}
	}
	return n
}

// subtokenize is misaki's SUBTOKEN_REGEX findall, by hand (Go's regexp has no lookahead):
//
//	^['‘’]+ | \p{Lu}(?=\p{Lu}\p{Ll}) | (?:^-)?(?:\d?[,.]?\d)+ | [-_]+ | ['‘’]{2,} |
//	\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}) | \p{L}+(?:['‘’]\p{L})* | [^-_\p{L}'‘’\d] | ['‘’]+$
func subtokenize(word string) []string {
	r := []rune(word)
	n := len(r)
	isQ := func(c rune) bool { return c == '\'' || c == '‘' || c == '’' }
	isD := func(c rune) bool { return c >= '0' && c <= '9' }
	isL, isLu, isLl := unicode.IsLetter, unicode.IsUpper, unicode.IsLower
	var out []string
	i := 0
	for i < n {
		// 1. ^['‘’]+
		if i == 0 && isQ(r[0]) {
			j := 0
			for j < n && isQ(r[j]) {
				j++
			}
			out = append(out, string(r[:j]))
			i = j
			continue
		}
		// 2. \p{Lu}(?=\p{Lu}\p{Ll})
		if isLu(r[i]) && i+2 < n && isLu(r[i+1]) && isLl(r[i+2]) {
			out = append(out, string(r[i]))
			i++
			continue
		}
		// 3. (?:^-)?(?:\d?[,.]?\d)+
		if j := matchNumber(r, i); j > i {
			out = append(out, string(r[i:j]))
			i = j
			continue
		}
		// 4. [-_]+
		if r[i] == '-' || r[i] == '_' {
			j := i
			for j < n && (r[j] == '-' || r[j] == '_') {
				j++
			}
			out = append(out, string(r[i:j]))
			i = j
			continue
		}
		// 5. ['‘’]{2,}
		if isQ(r[i]) && i+1 < n && isQ(r[i+1]) {
			j := i
			for j < n && isQ(r[j]) {
				j++
			}
			out = append(out, string(r[i:j]))
			i = j
			continue
		}
		// 6. lazy letters ending lower-case before an upper-case letter (camelCase)
		if isL(r[i]) {
			k, found := i, -1
			for k < n {
				if isL(r[k]) {
					if isLl(r[k]) && k+1 < n && isLu(r[k+1]) {
						found = k
						break
					}
					k++
				} else if isQ(r[k]) && k+1 < n && isL(r[k+1]) {
					k++
				} else {
					break
				}
			}
			if found >= 0 {
				out = append(out, string(r[i:found+1]))
				i = found + 1
				continue
			}
			// 7. \p{L}+(?:['‘’]\p{L})*
			j := i
			for j < n && isL(r[j]) {
				j++
			}
			for j+1 < n && isQ(r[j]) && isL(r[j+1]) {
				j += 2
				for j < n && isL(r[j]) {
					// (?:['‘’]\p{L})* takes one letter per group; further letters belong to the
					// next group only after another apostrophe, so stop here
					break
				}
			}
			out = append(out, string(r[i:j]))
			i = j
			continue
		}
		// 8. [^-_\p{L}'‘’\d]
		if !isQ(r[i]) && !isD(r[i]) {
			out = append(out, string(r[i]))
			i++
			continue
		}
		// 9. ['‘’]+$
		if isQ(r[i]) {
			j := i
			for j < n && isQ(r[j]) {
				j++
			}
			if j == n {
				out = append(out, string(r[i:j]))
				i = j
				continue
			}
		}
		i++ // no alternative matches here: findall moves on
	}
	return out
}

// matchNumber is (?:^-)?(?:\d?[,.]?\d)+ at i (greedy, with backtracking as the regex does).
func matchNumber(r []rune, i int) int {
	isD := func(c rune) bool { return c >= '0' && c <= '9' }
	j := i
	if i == 0 && j < len(r) && r[j] == '-' {
		j++
	}
	start := j
	for {
		k := j
		if k < len(r) && isD(r[k]) {
			// \d? then [,.]? then \d
			if k+1 < len(r) && isD(r[k+1]) {
				j = k + 2
				continue
			}
			if k+2 < len(r) && (r[k+1] == ',' || r[k+1] == '.') && isD(r[k+2]) {
				j = k + 3
				continue
			}
			j = k + 1 // \d? empty, [,.]? empty, \d = this digit
			continue
		}
		if k+1 < len(r) && (r[k] == ',' || r[k] == '.') && isD(r[k+1]) {
			j = k + 2
			continue
		}
		break
	}
	if j == start {
		return i
	}
	return j
}

// word groups: a single token, or the subtokens of one whitespace-free run
type group struct {
	one  *mtoken
	many []*mtoken
}

func retokenize(tokens []*mtoken) []group {
	var words []group
	currency := ""
	for i, token := range tokens {
		var tks []*mtoken
		if token.alias == nil && token.phonemes == nil {
			for _, t := range subtokenize(token.text) {
				tks = append(tks, &mtoken{text: t, tag: token.tag, isHead: true, numFlags: token.numFlags, stress: token.stress})
			}
		} else {
			tks = []*mtoken{token}
		}
		if len(tks) == 0 {
			continue
		}
		tks[len(tks)-1].ws = token.ws
		for j, tk := range tks {
			switch {
			case tk.alias != nil || tk.phonemes != nil:
			case tk.tag == "$" && currencies[tk.text] != [2]string{}:
				currency = tk.text
				tk.phonemes = sp("")
				tk.rating = 4
			case tk.tag == ":" && (tk.text == "-" || tk.text == "–"):
				tk.phonemes = sp("—")
				tk.rating = 3
			case punctTags[tk.tag] && !allAZ(tk.text):
				ph, ok := punctTagPh[tk.tag]
				if !ok {
					var b strings.Builder
					for _, c := range tk.text {
						if puncts[c] {
							b.WriteRune(c)
						}
					}
					ph = b.String()
				}
				tk.phonemes = sp(ph)
				tk.rating = 4
			case currency != "":
				if tk.tag != "CD" {
					currency = ""
				} else if j+1 == len(tks) && (i+1 == len(tokens) || tokens[i+1].tag != "CD") {
					tk.currency = currency
				}
			case 0 < j && j < len(tks)-1 && tk.text == "2" && isAlpha(lastRune(tks[j-1].text)+firstRune(tks[j+1].text)):
				tk.alias = sp("to")
			}
			if tk.alias != nil || tk.phonemes != nil {
				words = append(words, group{one: tk})
			} else if len(words) > 0 && words[len(words)-1].many != nil && words[len(words)-1].many[len(words[len(words)-1].many)-1].ws == "" {
				tk.isHead = false
				words[len(words)-1].many = append(words[len(words)-1].many, tk)
			} else if tk.ws != "" {
				words = append(words, group{one: tk})
			} else {
				words = append(words, group{many: []*mtoken{tk}})
			}
		}
	}
	for i, w := range words {
		if w.many != nil && len(w.many) == 1 {
			words[i] = group{one: w.many[0]}
		}
	}
	return words
}

func allAZ(s string) bool {
	for _, c := range strings.ToLower(s) {
		if c < 'a' || c > 'z' {
			return false
		}
	}
	return true
}

func lastRune(s string) string {
	r := []rune(s)
	if len(r) == 0 {
		return ""
	}
	return string(r[len(r)-1])
}

func firstRune(s string) string {
	r := []rune(s)
	if len(r) == 0 {
		return ""
	}
	return string(r[0])
}

func tokenContextOf(ctx tokenContext, ps *string, tk *mtoken) tokenContext {
	v := ctx.futureVowel
	if ps != nil && *ps != "" {
		for _, c := range *ps {
			if vowels[c] || consonants[c] || nonQuotePunct[c] {
				if nonQuotePunct[c] {
					v = nil
				} else {
					b := vowels[c]
					v = &b
				}
				break
			}
		}
	}
	ft := tk.text == "to" || tk.text == "To" || (tk.text == "TO" && (tk.tag == "TO" || tk.tag == "IN"))
	return tokenContext{futureVowel: v, futureTo: ft}
}

func resolveTokens(tokens []*mtoken) {
	var b strings.Builder
	for i, tk := range tokens {
		b.WriteString(tk.text)
		if i < len(tokens)-1 {
			b.WriteString(tk.ws)
		}
	}
	text := b.String()
	kinds := map[int]bool{}
	for _, c := range text {
		if subtokenJunks[c] {
			continue
		}
		switch {
		case unicode.IsLetter(c):
			kinds[0] = true
		case c >= '0' && c <= '9':
			kinds[1] = true
		default:
			kinds[2] = true
		}
	}
	prespace := strings.Contains(text, " ") || strings.Contains(text, "/") || len(kinds) > 1
	for i, tk := range tokens {
		if tk.phonemes == nil {
			if i == len(tokens)-1 && len([]rune(tk.text)) == 1 && nonQuotePunct[[]rune(tk.text)[0]] {
				tk.phonemes = sp(tk.text)
				tk.rating = 3
			} else if allIn(tk.text, subtokenJunks) {
				tk.phonemes = sp("")
				tk.rating = 3
			}
		} else if i > 0 {
			tk.prespace = prespace
		}
	}
	if prespace {
		return
	}
	type idx struct {
		hasP   bool
		weight int
		i      int
	}
	var indices []idx
	for i, tk := range tokens {
		if tk.phonemes != nil && *tk.phonemes != "" {
			indices = append(indices, idx{strings.ContainsRune(*tk.phonemes, primary), stressWeight(*tk.phonemes), i})
		}
	}
	if len(indices) == 2 && len([]rune(tokens[indices[0].i].text)) == 1 {
		i := indices[1].i
		tokens[i].phonemes = sp(applyStress(*tokens[i].phonemes, st(-0.5)))
		return
	}
	nP := 0
	for _, x := range indices {
		if x.hasP {
			nP++
		}
	}
	if len(indices) < 2 || nP <= (len(indices)+1)/2 {
		return
	}
	sort.Slice(indices, func(a, b int) bool {
		x, y := indices[a], indices[b]
		if x.hasP != y.hasP {
			return !x.hasP
		}
		if x.weight != y.weight {
			return x.weight < y.weight
		}
		return x.i < y.i
	})
	for _, x := range indices[:len(indices)/2] {
		tokens[x.i].phonemes = sp(applyStress(*tokens[x.i].phonemes, st(-0.5)))
	}
}

func allIn(s string, m map[rune]bool) bool {
	for _, c := range s {
		if !m[c] {
			return false
		}
	}
	return true
}

// Phonemize is G2P.__call__: the phoneme string for text.
func (g *G2P) Phonemize(text string) string {
	text = strings.TrimLeftFunc(text, unicode.IsSpace)
	raw := tokenizeText(text)
	tags := tagTokens(raw)
	tokens := make([]*mtoken, len(raw))
	for i, r := range raw {
		tokens[i] = &mtoken{text: r.text, tag: tags[i], ws: r.ws, isHead: true}
	}
	words := retokenize(tokens)
	ctx := tokenContext{}
	for wi := len(words) - 1; wi >= 0; wi-- {
		w := words[wi]
		if w.one != nil {
			tk := w.one
			if tk.phonemes == nil {
				if ps, r, ok := g.Lex.word(tk, &ctx); ok {
					tk.phonemes, tk.rating = sp(ps), r
				}
			}
			if tk.phonemes == nil && g.Fallback != nil {
				if ps, ok := g.Fallback(tk.text); ok {
					tk.phonemes, tk.rating = sp(ps), 2
				}
			}
			ctx = tokenContextOf(ctx, tk.phonemes, tk)
			continue
		}
		ws := w.many
		left, right := 0, len(ws)
		shouldFallback := false
		for left < right {
			var tk *mtoken
			busy := false
			for _, x := range ws[left:right] {
				if x.alias != nil || x.phonemes != nil {
					busy = true
					break
				}
			}
			if !busy {
				tk = mergeTokens(ws[left:right], nil)
			}
			var ps string
			var rating int
			ok := false
			if tk != nil {
				ps, rating, ok = g.Lex.word(tk, &ctx)
			}
			if ok {
				ws[left].phonemes = sp(ps)
				ws[left].rating = rating
				for _, x := range ws[left+1 : right] {
					x.phonemes = sp("")
					x.rating = rating
				}
				ctx = tokenContextOf(ctx, sp(ps), tk)
				right = left
				left = 0
			} else if left+1 < right {
				left++
			} else {
				right--
				t := ws[right]
				if t.phonemes == nil {
					if allIn(t.text, subtokenJunks) {
						t.phonemes = sp("")
						t.rating = 3
					} else if g.Fallback != nil {
						shouldFallback = true
						break
					}
				}
				left = 0
			}
		}
		if shouldFallback {
			tk := mergeTokens(ws, nil)
			if ps, ok := g.Fallback(tk.text); ok {
				ws[0].phonemes, ws[0].rating = sp(ps), 2
			} else {
				ws[0].phonemes = nil
			}
			for j := 1; j < len(ws); j++ {
				ws[j].phonemes = sp("")
				ws[j].rating = ws[0].rating
			}
		} else {
			resolveTokens(ws)
		}
	}
	var b strings.Builder
	unk := g.Unk
	for _, w := range words {
		tk := w.one
		if w.many != nil {
			tk = mergeTokens(w.many, &unk)
		}
		ps := unk
		if tk.phonemes != nil {
			ps = strings.NewReplacer("ɾ", "T", "ʔ", "t").Replace(*tk.phonemes)
		}
		b.WriteString(ps)
		b.WriteString(tk.ws)
	}
	return b.String()
}
