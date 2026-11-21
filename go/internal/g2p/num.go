package g2p

import (
	"regexp"
	"strconv"
	"strings"
)

// num2words (English), the parts misaki uses: cardinal, ordinal, year, and a float's
// digits after the point.

var (
	onesW  = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	tensW  = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
	scales = []struct {
		v    int64
		name string
	}{{1_000_000_000_000, "trillion"}, {1_000_000_000, "billion"}, {1_000_000, "million"}, {1000, "thousand"}}
)

func cardinal(n int64) string {
	if n < 0 {
		return "minus " + cardinal(-n)
	}
	switch {
	case n < 20:
		return onesW[n]
	case n < 100:
		if n%10 == 0 {
			return tensW[n/10]
		}
		return tensW[n/10] + "-" + onesW[n%10]
	case n < 1000:
		s := onesW[n/100] + " hundred"
		if r := n % 100; r > 0 {
			s += " and " + cardinal(r)
		}
		return s
	}
	for _, sc := range scales {
		if n >= sc.v {
			s := cardinal(n/sc.v) + " " + sc.name
			if r := n % sc.v; r > 0 {
				if r < 100 {
					s += " and " + cardinal(r)
				} else {
					s += ", " + cardinal(r)
				}
			}
			return s
		}
	}
	return strconv.FormatInt(n, 10)
}

var ordinalW = map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

func ordinal(n int64) string {
	c := cardinal(n)
	i := strings.LastIndexAny(c, " -")
	head, last := c[:i+1], c[i+1:]
	if o, ok := ordinalW[last]; ok {
		return head + o
	}
	if strings.HasSuffix(last, "y") {
		return head + last[:len(last)-1] + "ieth"
	}
	return head + last + "th"
}

func year(n int64) string {
	high, low := n/100, n%100
	if high == 0 || (high%10 == 0 && low < 10) || high >= 100 {
		return cardinal(n)
	}
	h := cardinal(high)
	switch {
	case low == 0:
		return h + " hundred"
	case low < 10:
		return h + " oh-" + cardinal(low)
	}
	return h + " " + cardinal(low)
}

// floatWords is num2words(float(s)): the integer part, then "point" and each digit.
func floatWords(s string) string {
	ip, fp, _ := strings.Cut(s, ".")
	n, _ := strconv.ParseInt(ip, 10, 64)
	fp = strings.TrimRight(fp, "0")
	if fp == "" {
		return cardinal(n)
	}
	w := cardinal(n) + " point"
	for _, d := range fp {
		w += " " + onesW[d-'0']
	}
	return w
}

var (
	nonAZ       = regexp.MustCompile(`[^a-z]+`)
	numSuffixRe = regexp.MustCompile(`[a-z']+$`)
)

type phr struct {
	ps     string
	rating int
}

// getNumber is Lexicon.get_number.
func (l *Lexicon) getNumber(word, currency string, isHead bool, numFlags string) (string, int, bool) {
	suffix := numSuffixRe.FindString(word)
	word = word[:len(word)-len(suffix)]
	var result []phr
	look := func(w string, stress Stress) {
		if ps, r, ok := l.lookup(w, "", stress, nil); ok {
			result = append(result, phr{ps, r})
		} else {
			result = append(result, phr{"", 0}) // Python appends (None, None): the join fails
		}
	}
	if strings.HasPrefix(word, "-") {
		look("minus", nil)
		word = word[1:]
	}
	extend := func(text string, first bool) {
		splits := nonAZ.Split(text, -1)
		for i, w := range splits {
			if w != "and" || strings.Contains(numFlags, "&") {
				if first && i == 0 && len(splits) > 1 && w == "one" && strings.Contains(numFlags, "a") {
					result = append(result, phr{"ə", 4})
				} else {
					var s Stress
					if w == "point" {
						s = st(-2)
					}
					look(w, s)
				}
			} else if w == "and" && strings.Contains(numFlags, "n") && len(result) > 0 {
				result[len(result)-1].ps += "ən"
			}
		}
	}
	extendNum := func(num string, first bool) { // extend_num(num) without escape: num2words(int(num))
		n, _ := strconv.ParseInt(num, 10, 64)
		extend(cardinal(n), first)
	}
	_, isCur := currencies[currency]
	switch {
	case isDigitStr(word) && ordinals[suffix]:
		n, _ := strconv.ParseInt(word, 10, 64)
		extend(ordinal(n), true)
	case len(result) == 0 && len(word) == 4 && !isCur && isDigitStr(word):
		n, _ := strconv.ParseInt(word, 10, 64)
		extend(year(n), true)
	case !isHead && !strings.Contains(word, "."):
		num := strings.ReplaceAll(word, ",", "")
		switch {
		case num == "":
		case num[0] == '0' || len(num) > 3:
			for _, d := range num {
				extendNum(string(d), false)
			}
		case len(num) == 3 && !strings.HasSuffix(num, "00"):
			extendNum(num[:1], true)
			if num[1] == '0' {
				look("O", st(-2))
				extendNum(num[2:], false)
			} else {
				extendNum(num[1:], false)
			}
		default:
			extendNum(num, true)
		}
	case strings.Count(word, ".") > 1 || !isHead:
		first := true
		for _, num := range strings.Split(strings.ReplaceAll(word, ",", ""), ".") {
			if num == "" {
			} else if num[0] == '0' || (len(num) != 2 && strings.Trim(num[1:], "0") != "") {
				for _, d := range num {
					extendNum(string(d), false)
				}
			} else {
				extendNum(num, first)
			}
			first = false
		}
	case isCur && isCurrencyAmount(word):
		units := currencies[currency]
		parts := strings.Split(strings.ReplaceAll(word, ",", ""), ".")
		type pair struct {
			n    int64
			unit string
		}
		var pairs []pair
		for i, p := range parts {
			if i >= 2 {
				break
			}
			n, _ := strconv.ParseInt(p, 10, 64)
			pairs = append(pairs, pair{n, units[i]})
		}
		if len(pairs) > 1 {
			if pairs[1].n == 0 {
				pairs = pairs[:1]
			} else if pairs[0].n == 0 {
				pairs = pairs[1:]
			}
		}
		for i, p := range pairs {
			if i > 0 {
				look("and", nil)
			}
			extend(cardinal(p.n), i == 0)
			if (p.n != 1 && p.n != -1) && p.unit != "pence" {
				if ps, r, ok := l.stemS(p.unit+"s", "", nil, nil); ok {
					result = append(result, phr{ps, r})
				} else {
					result = append(result, phr{"", 0})
				}
			} else {
				look(p.unit, nil)
			}
		}
	default:
		var text string
		switch {
		case isDigitStr(word):
			n, _ := strconv.ParseInt(word, 10, 64)
			text = cardinal(n)
		case !strings.Contains(word, "."):
			n, _ := strconv.ParseInt(strings.ReplaceAll(word, ",", ""), 10, 64)
			if ordinals[suffix] {
				text = ordinal(n)
			} else {
				text = cardinal(n)
			}
		default:
			w := strings.ReplaceAll(word, ",", "")
			if w[0] == '.' {
				text = "point"
				for _, d := range w[1:] {
					text += " " + onesW[d-'0']
				}
			} else {
				text = floatWords(w)
			}
		}
		extend(text, true)
	}
	if len(result) == 0 {
		return "", 0, false
	}
	parts := make([]string, len(result))
	rating := 99
	for i, p := range result {
		if p.rating == 0 {
			return "", 0, false
		}
		parts[i] = p.ps
		if p.rating < rating {
			rating = p.rating
		}
	}
	ps := strings.Join(parts, " ")
	switch suffix {
	case "s", "'s":
		return l.suffS(ps), rating, true
	case "ed", "'d":
		return l.suffEd(ps), rating, true
	case "ing":
		return l.suffIng(ps), rating, true
	}
	return ps, rating, true
}

func isCurrencyAmount(word string) bool {
	if !strings.Contains(word, ".") {
		return true
	}
	if strings.Count(word, ".") > 1 {
		return false
	}
	_, cents, _ := strings.Cut(word, ".")
	return len(cents) < 3
}
