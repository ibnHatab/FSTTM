package g2p

import (
	"regexp"
	"strings"
	"unicode"
)

// The tokenizer and tagger that stand in for spaCy's en_core_web_sm: they split text the
// way spaCy's English tokenizer does on everyday sentences (punctuation off words,
// contractions, hyphens between letters, $ and % off numbers, abbreviations kept) and give
// each token a Penn Treebank tag, guessed by rules, as far as misaki's rules read tags:
// DT for determiners ("a" → ɐ), PRP ("I"), IN/RP/RB ("in" stressed or not, "by"), TO,
// NNP (capitalized names, "AM"/"PM" after a time), CD, $, the punctuation tags, and the
// noun / verb / adjective / past-tense guesses the lexicon's heteronyms are keyed by
// ("read", "close", "live", "record", "use", "used to").

type rawTok struct {
	text, ws string
}

var (
	abbrevRe   = regexp.MustCompile(`^(?:[A-Za-z]\.){2,}$|^[A-Z]\.$`)
	abbrevs    = map[string]bool{"Mr.": true, "Mrs.": true, "Ms.": true, "Dr.": true, "St.": true, "Jr.": true, "Sr.": true, "vs.": true, "etc.": true, "e.g.": true, "i.e.": true, "a.m.": true, "p.m.": true, "No.": true}
	numericRe  = regexp.MustCompile(`^[-+]?\d+(?:[.,:]\d+)*(?:st|nd|rd|th|s)?$`)
	prefixChar = "\"'“‘([{$€£#¿¡"
	suffixChar = "\"'”’)]}.,!?;:%…"
)

var contractions = []string{"n't", "'s", "'m", "'re", "'ve", "'ll", "'d", "’s", "’m", "’re", "’ve", "’ll", "’d", "n’t"}

// splitToken is spaCy's prefix / suffix / infix loop on one whitespace-free chunk.
func splitToken(chunk string) []string {
	if chunk == "" {
		return nil
	}
	var pre, suf []string
	s := chunk
	for s != "" {
		if abbrevs[s] || abbrevRe.MatchString(s) || numericRe.MatchString(s) {
			break
		}
		r := []rune(s)
		if len(r) > 1 && strings.ContainsRune(prefixChar, r[0]) {
			pre = append(pre, string(r[0]))
			s = string(r[1:])
			continue
		}
		if strings.HasSuffix(s, "...") && len(s) > 3 {
			suf = append([]string{"..."}, suf...)
			s = s[:len(s)-3]
			continue
		}
		last := r[len(r)-1]
		if len(r) > 1 && strings.ContainsRune(suffixChar, last) {
			// a word's own contraction is split below, a closing quote/punctuation here
			suf = append([]string{string(last)}, suf...)
			s = string(r[:len(r)-1])
			continue
		}
		break
	}
	var mid []string
	if s != "" {
		mid = splitContraction(s)
	}
	out := append(pre, mid...)
	return append(out, suf...)
}

// specialSplits are spaCy's tokenizer exceptions for the contractions whose first half
// changes ("can't" → "ca" "n't").
var specialSplits = map[string][]string{
	"can't": {"ca", "n't"}, "won't": {"wo", "n't"}, "shan't": {"sha", "n't"}, "Can't": {"Ca", "n't"}, "Won't": {"Wo", "n't"},
	"cannot": {"can", "not"}, "Cannot": {"Can", "not"}, "gonna": {"gon", "na"}, "gotta": {"got", "ta"}, "wanna": {"wan", "na"},
	"let's": {"let", "'s"}, "Let's": {"Let", "'s"},
}

func splitContraction(s string) []string {
	if parts, ok := specialSplits[s]; ok {
		return parts
	}
	if abbrevs[s] || abbrevRe.MatchString(s) || numericRe.MatchString(s) {
		return []string{s}
	}
	ls := strings.ToLower(s)
	for _, c := range contractions {
		if strings.HasSuffix(ls, c) && len(s) > len(c) {
			head := s[:len(s)-len(c)]
			if isAlpha(head) {
				return []string{head, s[len(s)-len(c):]}
			}
		}
	}
	return splitInfix(s)
}

// splitInfix splits hyphens between letters ("Wi-Fi" → "Wi" "-" "Fi"), and "/" between words.
func splitInfix(s string) []string {
	r := []rune(s)
	var out []string
	start := 0
	for i := 1; i < len(r)-1; i++ {
		if (r[i] == '-' || r[i] == '/') && unicode.IsLetter(r[i-1]) && unicode.IsLetter(r[i+1]) {
			out = append(out, string(r[start:i]), string(r[i]))
			start = i + 1
		}
	}
	return append(out, string(r[start:]))
}

// tokenizeText splits on whitespace (keeping each token's following whitespace) and then
// within each chunk.
func tokenizeText(text string) []rawTok {
	var out []rawTok
	r := []rune(text)
	i := 0
	for i < len(r) {
		if unicode.IsSpace(r[i]) {
			i++
			continue
		}
		j := i
		for j < len(r) && !unicode.IsSpace(r[j]) {
			j++
		}
		k := j
		for k < len(r) && unicode.IsSpace(r[k]) {
			k++
		}
		parts := splitToken(string(r[i:j]))
		for pi, p := range parts {
			ws := ""
			if pi == len(parts)-1 && j < len(r) {
				ws = string(r[j:k])
				if len([]rune(ws)) > 1 { // spaCy keeps one space as whitespace, more as tokens
					ws = " "
				}
			}
			out = append(out, rawTok{p, ws})
		}
		i = k
	}
	return out
}

var (
	determiners = set2("the", "a", "an", "this", "that", "these", "those", "no", "every", "each", "some", "any", "all", "another", "either", "neither", "both")
	pronouns    = set2("i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "yourself", "itself", "ourselves", "themselves")
	possessives = set2("my", "your", "its", "our", "their", "his")
	modals      = set2("can", "ca", "could", "will", "wo", "would", "shall", "sha", "should", "may", "might", "must")
	beForms     = set2("is", "was", "are", "were", "be", "been", "being", "am", "'s", "'re", "'m", "seems", "seem", "looks", "feels", "stays", "get", "gets", "got")
	preps       = set2("for", "at", "of", "with", "until", "from", "into", "about", "without", "after", "before", "during", "under", "over", "through", "between", "near", "since", "like", "than", "per", "via", "behind", "above", "below", "across", "around", "toward", "towards", "upon", "within")
	particles   = set2("on", "off", "up", "down", "out", "in", "back", "away")
	conjs       = set2("and", "or", "but", "nor", "yet", "so")
	adverbs     = set2("not", "n't", "already", "now", "here", "there", "then", "again", "still", "only", "just", "also", "very", "too", "yet", "soon", "once", "ahead", "else", "never", "always", "often", "outside", "inside", "home", "first")
	interjects  = set2("please", "hello", "hi", "hey", "ok", "okay", "sorry", "yes", "no", "hmm", "oh", "thanks", "sure", "well")
	whWords     = set2("what", "who", "whom", "which", "whose")
	whAdv       = set2("when", "where", "why", "how")
	complementV = set2("know", "knows", "think", "say", "says", "said", "sure", "check", "make", "seems", "means", "hope", "believe", "glad", "notice", "noticed", "see", "tell", "told")
	timeWords   = set2("am", "pm", "a.m.", "p.m.")
	nameStarts  = set2("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")
	pastMarkers = set2("yesterday", "ago", "earlier", "last", "already")
)

func set2(ws ...string) map[string]bool {
	m := map[string]bool{}
	for _, w := range ws {
		m[w] = true
	}
	return m
}

func punctTag(t string) (string, bool) {
	switch t {
	case ".", "!", "?", "...", "…":
		return ".", true
	case ",":
		return ",", true
	case ":", ";", "—", "–", "--":
		return ":", true
	case "-":
		return "HYPH", true
	case "(", "[", "{":
		return "-LRB-", true
	case ")", "]", "}":
		return "-RRB-", true
	case "$", "€", "£":
		return "$", true
	case "#":
		return "#", true
	}
	return "", false
}

// tagTokens gives each token a tag.
func tagTokens(toks []rawTok) []string {
	tags := make([]string, len(toks))
	quoteOpen := false
	lower := func(i int) string {
		if i < 0 || i >= len(toks) {
			return ""
		}
		return strings.ToLower(toks[i].text)
	}
	sentStart := func(i int) bool {
		for k := i - 1; k >= 0; k-- {
			switch toks[k].text {
			case "\"", "“", "‘", "'", "(":
				continue
			case ".", "!", "?", ":", ";":
				return true
			}
			return false
		}
		return true
	}
	// first pass: the closed classes
	for i, tk := range toks {
		t, lw := tk.text, strings.ToLower(tk.text)
		switch {
		case t == "\"" || t == "“" || t == "”":
			if t == "“" || (t == "\"" && !quoteOpen) {
				tags[i] = "``"
				quoteOpen = true
			} else {
				tags[i] = "''"
				quoteOpen = false
			}
		case func() bool { p, ok := punctTag(t); tags[i] = p; return ok }():
		case t == "%":
			tags[i] = "NN"
		case numericRe.MatchString(t) && !strings.HasSuffix(lw, "st") && !strings.HasSuffix(lw, "nd") && !strings.HasSuffix(lw, "rd") && !strings.HasSuffix(lw, "th"):
			tags[i] = "CD"
		case numericRe.MatchString(t):
			tags[i] = "JJ"
		case lw == "million" || lw == "billion" || lw == "thousand" || lw == "hundred":
			tags[i] = "CD"
		case lw == "to":
			tags[i] = "TO"
		case t == "I":
			tags[i] = "PRP"
		case lw == "n't" || lw == "n’t" || lw == "not":
			tags[i] = "RB"
		case lw == "'s" || lw == "’s":
			if i > 0 && (pronouns[lower(i-1)] || lower(i-1) == "that" || lower(i-1) == "there" || lower(i-1) == "what" || lower(i-1) == "let") {
				tags[i] = "VBZ"
			} else {
				tags[i] = "POS"
			}
		case lw == "'m" || lw == "'re" || lw == "'ve":
			tags[i] = "VBP"
		case lw == "'ll" || lw == "'d":
			tags[i] = "MD"
		case modals[lw] && !(lw == "may" && t == "May"):
			tags[i] = "MD"
		case possessives[lw]:
			tags[i] = "PRP$"
		case lw == "that":
			tags[i] = "DT"
			if i > 0 && complementV[lower(i-1)] {
				tags[i] = "IN"
			} else if i > 0 && i+1 < len(toks) && isNounLike(toks[i-1].text) && verbish(lower(i+1)) {
				tags[i] = "WDT" // a relative "that" after a noun ("a rule that switches")
			}
		case lw == "there" && i+1 < len(toks) && beForms[lower(i+1)]:
			tags[i] = "EX"
		case determiners[lw]:
			tags[i] = "DT"
		case pronouns[lw]:
			tags[i] = "PRP"
		case lw == "her":
			tags[i] = "PRP$"
		case conjs[lw] && lw != "yet":
			tags[i] = "CC"
		case whWords[lw]:
			tags[i] = "WP"
		case whAdv[lw]:
			tags[i] = "WRB"
		case interjects[lw] && (sentStart(i) || i+1 < len(toks) && toks[i+1].text == ","):
			tags[i] = "UH"
		case lw == "if" || lw == "because" || lw == "while" || lw == "though" || lw == "although" || lw == "unless" || lw == "whether":
			tags[i] = "IN"
		case preps[lw]:
			tags[i] = "IN"
		}
	}
	// second pass: the open classes, from the neighbours
	for i, tk := range toks {
		if tags[i] != "" {
			continue
		}
		t, lw := tk.text, strings.ToLower(tk.text)
		prev, next := "", ""
		if i > 0 {
			prev = tags[i-1]
		}
		if i+1 < len(tags) {
			next = tags[i+1]
		}
		pl := lower(i - 1)
		switch {
		case timeWords[lw] && i > 0 && (prev == "CD" || numericRe.MatchString(toks[i-1].text)):
			tags[i] = "NNP"
		case lw == "by":
			if next == "DT" || next == "PRP$" || next == "CD" || next == "PRP" || next == "NNP" {
				tags[i] = "IN"
			} else {
				tags[i] = "RB"
			}
		case particles[lw]:
			switch {
			case next == "DT" || next == "PRP$" || next == "CD" || next == "PRP" || (i+1 < len(toks) && nameStarts[lower(i+1)]):
				tags[i] = "IN"
			case strings.HasPrefix(prev, "VB"):
				tags[i] = "RP"
			default:
				tags[i] = "RB"
			}
			if lw == "in" && tags[i] != "IN" && (next == "NN" || next == "JJ" || next == "") {
				tags[i] = "IN"
			}
		case adverbs[lw]:
			tags[i] = "RB"
		case t != lw && !sentStart(i) && unicode.IsUpper([]rune(t)[0]):
			tags[i] = "NNP"
		case nameStarts[lw]:
			tags[i] = "NNP"
		case pl == "to" || prev == "MD" || (prev == "RB" && (i >= 2 && tags[i-2] == "MD")) || (sentStart(i) && prev != "UH" && i+1 < len(toks)) || prev == "UH":
			tags[i] = "VB"
		case prev == "PRP":
			if pastMarkers[lower(i+1)] || pastMarkers[lower(i+2)] || pastMarkers[lower(i+3)] {
				tags[i] = "VBD"
			} else if strings.HasSuffix(lw, "ed") {
				tags[i] = "VBD"
			} else {
				tags[i] = "VBP"
			}
		case prev == "DT" || prev == "PRP$" || prev == "JJ" || prev == "CD" || prev == "POS" || prev == "IN":
			tags[i] = "NN"
		case beForms[pl]:
			tags[i] = "JJ"
		default:
			tags[i] = "NN"
		}
		if lw == "used" && lower(i+1) == "to" {
			tags[i] = "VBD"
		}
	}
	return tags
}

// misaki's NNP for "AM"/"PM": the tagger above; its noun tag for a bare "%": NN.

// isNounLike: a word the first pass left untagged (an open-class word) that a relative
// "that" can follow; the closed classes are not.
func isNounLike(w string) bool {
	lw := strings.ToLower(w)
	if !isAlpha(w) || determiners[lw] || pronouns[lw] || possessives[lw] || modals[lw] || preps[lw] || conjs[lw] || beForms[lw] || complementV[lw] || adverbs[lw] || particles[lw] || interjects[lw] {
		return false
	}
	return true
}

// verbish: the word after a relative "that" ("a rule that switches", "that is", "that can").
func verbish(lw string) bool {
	if beForms[lw] || modals[lw] {
		return true
	}
	return isAlpha(lw) && len(lw) > 3 && (strings.HasSuffix(lw, "es") || strings.HasSuffix(lw, "ed") || (strings.HasSuffix(lw, "s") && !strings.HasSuffix(lw, "ss")))
}
