// tts.js — speechSynthesis as the dialog needs it: a reply is a list of
// sentences spoken in a chain (the floor stays SYSTEM across them), the
// sentence being spoken is known (echo rejection), and on a cut the text
// actually heard is known from onboundary's charIndex (the heard-text
// invariant, the README's Design). Linux speech-dispatcher voices fire no
// onboundary at all (measured, Chrome 154): there the heard text is
// estimated from the time spoken, at the rate the finished utterances
// played, and marked estimated.

export class BrowserTTS {
  constructor(synth = globalThis.speechSynthesis, { voice = null, rate = 1 } = {}) {
    this.synth = synth;
    this.voice = voice;
    this.rate = rate;
    this.cur = null; // {sentences, i, char, onend, startedAt}
    this.charsPerMs = 0.014; // ~14 chars/s at rate 1, re-measured on every completed utterance
    this.boundaries = false; // has this voice ever fired onboundary
    this.estimated = false;  // was the last heard() an estimate
  }

  get speaking() { return this.cur !== null; }
  get sentence() { return this.cur ? this.cur.sentences[this.cur.i] : ''; }

  // speak plays the sentences in order; hooks: onstart(i), onend(completed)
  speak(sentences, { onstart, onend } = {}) {
    this.cancel();
    const cur = { sentences, i: 0, char: 0, onend };
    this.cur = cur;
    const next = () => {
      if (this.cur !== cur) return;
      if (cur.i >= sentences.length) { this.cur = null; onend?.(true); return; }
      const u = new SpeechSynthesisUtterance(sentences[cur.i]);
      if (this.voice) u.voice = this.voice;
      u.rate = this.rate;
      cur.char = 0;
      u.onstart = () => { cur.startedAt = performance.now(); onstart?.(cur.i); };
      u.onboundary = e => { this.boundaries = true; if (this.cur === cur) cur.char = e.charIndex; };
      u.onend = () => {
        if (this.cur !== cur) return;
        const ms = performance.now() - (cur.startedAt ?? performance.now());
        if (ms > 300) this.charsPerMs = sentences[cur.i].length / ms;
        cur.i++; next();
      };
      u.onerror = () => { if (this.cur !== cur) return; cur.i++; next(); };
      this.synth.speak(u);
    };
    next();
  }

  // heard is what the listener heard so far: whole sentences before the
  // current one, and the current one up to the word being spoken
  heard() {
    const c = this.cur;
    if (!c) return null;
    const done = c.sentences.slice(0, c.i).join(' ');
    const s = c.sentences[c.i] ?? '';
    let char = c.char;
    this.estimated = !this.boundaries;
    if (this.estimated) { // no boundaries from this voice: by time, cut back to a word end
      char = c.startedAt ? Math.min(s.length, Math.round((performance.now() - c.startedAt) * this.charsPerMs)) : 0;
      char = char >= s.length ? s.length : Math.max(0, s.lastIndexOf(' ', char));
    }
    const part = s.slice(0, char).trim();
    return [done, part].filter(Boolean).join(' ');
  }

  // cancel stops at once; returns the heard text (null when silent)
  cancel() {
    const c = this.cur;
    if (!c) return null;
    const h = this.heard();
    this.cur = null;
    this.synth.cancel();
    c.onend?.(false);
    return h;
  }
}

// splitSentences cuts a reply for the chain: on . ? ! with at least min
// characters per piece, so a short "Okay." is not its own
// utterance unless it is the whole reply.
export function splitSentences(text, min = 40) {
  const parts = text.match(/[^.?!]+[.?!]+["')\]]*\s*|[^.?!]+$/g) ?? [text];
  const out = [];
  let acc = '';
  for (const p of parts) {
    acc += p;
    if (acc.trim().length >= min) { out.push(acc.trim()); acc = ''; }
  }
  if (acc.trim()) out.length ? (out[out.length - 1] += ' ' + acc.trim()) : out.push(acc.trim());
  return out;
}
