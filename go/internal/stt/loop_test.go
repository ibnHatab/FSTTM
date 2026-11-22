package stt

import (
	"strings"
	"testing"
)

func TestLooped(t *testing.T) {
	loops := []string{
		"The measures are range error per source, stop gap with and without the stereo channel, " +
			strings.Repeat("and without the stereo channel, ", 18) + "and whether the track survives the sector exit.",
		"and her " + strings.Repeat("her ", 40),
		strings.Repeat("I think it's a great story.", 12),
		strings.Repeat("I'm not sure if you're interested in this, but I'm not sure.", 4),
		"[Music]" + strings.Repeat("[Music]", 30),
	}
	for _, s := range loops {
		if !Looped(s) {
			t.Errorf("not seen as a loop: %.60q", s)
		}
	}
	speech := []string{
		"Hey Nina, turn on the lab ventilation.",
		"No no no, the other one.",
		"Turn it off, turn it off now.",
		"Acceptance once I pass the pages is the two existing cards plus a new range card, a person and a box at 1.4 m in the stereo sector, and one walking out of it.",
		"Yes, and explain the coordinate system should be translated in CARLA, not to set on the axis.",
		strings.Repeat("And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. ", 3),
		"It is 26.9 degrees Celsius outside the bedroom.",
	}
	for _, s := range speech {
		if Looped(s) {
			t.Errorf("speech seen as a loop: %q", s)
		}
	}
}

func TestCollapse(t *testing.T) {
	in := "stop gap with and without the stereo channel, " + strings.Repeat("and without the stereo channel, ", 18) + "and whether the track survives."
	got := Collapse(in)
	if Looped(got) || strings.Count(got, "stereo channel") > 2 || !strings.HasSuffix(got, "and whether the track survives.") {
		t.Fatalf("collapse: %q", got)
	}
	if g := Collapse("her her her her her her her her"); g != "her …" {
		t.Fatalf("collapse words: %q", g)
	}
}

func TestCtxFor(t *testing.T) {
	cases := []struct {
		floor uint
		secs  float64
		want  uint
	}{
		{768, 2.5, 768}, {768, 10, 768}, {768, 15, 832}, {768, 17.7, 960}, {768, 25, 1344}, {768, 28.7, 0}, {768, 60, 0}, {0, 5, 0},
	}
	for _, c := range cases {
		if g := CtxFor(c.floor, int(c.secs*16000)); g != c.want {
			t.Errorf("CtxFor(%d, %.1fs) = %d, want %d", c.floor, c.secs, g, c.want)
		}
	}
}
