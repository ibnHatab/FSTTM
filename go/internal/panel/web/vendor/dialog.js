// dialog.js — the dialog agent in the browser (the README's Design): browser
// signals → FSTTM actions, attention in front, barge-in by policy, textual
// echo rejection, and the heard-text record of every system turn.
//
// It is pure logic over injected parts, so the event table (web/eventtable.js) is verified with
// scripted signals in node and in Chrome alike; attachRecognizer() wires a
// real SpeechRecognition to the same entry points.
//
//   recognizer → speechStart() · interim(text) · final(text)
//   tts        → BrowserTTS-like: speak(sentences,{onstart,onend}) · cancel() → heard · sentence · speaking
//   llm        → async (text, {signal, id}) → {voice, firstMs}; id names the turn
//   onHeard    → (id, heard): what of turn id was heard, on every change (a
//                cut, the turn heard whole, a resume finished) — the backend
//                adapter's cut(id, heard); say(…, id) gives any utterance an id
//   log        → bench.js (kind, fields)

import { splitSentences } from './tts.js';

export const STOP_WORDS = ['stop', 'halt', 'freeze', 'quiet'];

// RESUME_PHRASES pick an interrupted narration up where it was cut, without
// a new LLM call (the Python FSTTM server's RESUME_PHRASES).
export const RESUME_PHRASES = ['continue', 'go on', 'please continue', 'keep going', 'go ahead',
  'please go on', 'carry on', 'and then', 'what else'];
const resumePhrase = text => RESUME_PHRASES.includes(text.toLowerCase().replace(/[^a-z' ]+/g, ' ').trim().replace(/\s+/g, ' '));

// keywordStop: at most three words, one of them a stop word (the harness's
// F12 rule; here it also runs on interims).
export function keywordStop(text, words = STOP_WORDS) {
  const w = text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
  return w.length > 0 && w.length <= 3 && w.some(x => words.includes(x));
}

const toks = s => (s.toLowerCase().match(/[a-z0-9']+/g) ?? []);

// The echo measure. The recognizer re-spells what it hears of her: digits
// as words ("countertop 3" → "counter top three"), compounds split
// ("ALFWorld" → "Alf World"), near-misses ("household" → "the house"), so
// tokens never compare. Both sides are normalized (numbers 0–99 spelled
// out, letters only, no spaces, a doubled letter in a word single: "Alright"
// heard as "All right") and compared by character trigrams: the
// share of the transcript's trigrams found in the spoken text's.
// ECHO is measured on the demo's echoes (test/echo.mjs): its echoes share
// 0.78–1.00, the user's words over the same speech at most 0.73.
export const ECHO = 0.75;
const ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(' ');
const TENS = ',,twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety'.split(',');
const spell = n => n < 20 ? ONES[n] : n < 100 ? TENS[Math.floor(n / 10)] + (n % 10 ? ONES[n % 10] : '')
  : String(n).split('').map(d => ONES[+d]).join('');
export const norm = s => (s.toLowerCase().replace(/\d+/g, m => ` ${spell(+m)} `).match(/[a-z]+/g) ?? [])
  .map(w => w.replace(/(.)\1+/g, '$1')).join('');
const grams = s => { const g = new Set(); for (let i = 0; i + 3 <= s.length; i++) g.add(s.slice(i, i + 3)); return g; };
const found = (g, h) => { let n = 0; for (const x of g) if (h.has(x)) n++; return n / g.size; };
const units = spoken => (Array.isArray(spoken) ? spoken : [spoken]).filter(Boolean);

// windows: what she said as the recognizer may hear it in one final — one
// sentence, or two neighbours (a final spans two short lines at most) — so
// the words of a long backlog said apart do not add up to an echo
function* windows(spoken) {
  const us = units(spoken);
  for (let i = 0; i < us.length; i++) for (let k = 1; k <= 2 && i + k <= us.length; k++) yield norm(us.slice(i, i + k).join(' '));
}

// echoShare: the transcript's best share of trigrams in a window of what
// she said; spoken is the sentences in the order said (a string is one).
// Under three letters a transcript is echo when its letters are in a window.
export function echoShare(text, spoken) {
  const t = norm(text);
  if (!t) return 0;
  const g = grams(t);
  let best = 0;
  for (const w of windows(spoken)) best = Math.max(best, g.size ? found(g, grams(w)) : w.includes(t) ? 1 : 0);
  return best;
}

// pastEcho is what the user said, for the LLM: the transcript after its
// echo. Echo comes first in a transcript (the recognizer hears her, then
// the user). Within a window, a word is hers when every trigram it adds to
// the transcript so far — its own and the two across the join, which carry
// the order — is found there (one may be there by chance: "what a lovely"
// holds the "ata" of "what about"); the echo ends at the last two of hers
// in a row whose prefix still measures ECHO and holds minLetters (short
// runs — "go to", "put the" — are everyone's), before any three words in a
// row that are not hers ("…experiment can we continue to explore how
// different": her words go on after the user's, Chrome 2026-10-24). So one misheard word inside
// an echo stays in it, and words she used apart are the user's: "what about
// jupiter" after a reply with "what" and "about" apart reaches the LLM whole.
export function pastEcho(text, spoken, minLetters = 8) {
  let words = text.trim().split(/\s+/).filter(Boolean), end;
  do { // an interim that grew across a long reply holds one echo after another
    end = echoEnd(words, spoken, minLetters);
    words = words.slice(end + 1);
  } while (end >= 0 && words.length);
  return words.join(' ');
}

function echoEnd(words, spoken, minLetters) {
  let end = -1;
  for (const sp of windows(spoken)) {
    const h = grams(sp);
    let acc = '';
    const hers = words.map(w => {
      const prev = acc; acc += norm(w);
      if (acc === prev) return true; // no letters: goes with its neighbours
      if (acc.length < 3) return sp.includes(acc);
      const add = grams(acc); for (const x of grams(prev)) add.delete(x);
      return add.size === 0 || found(add, h) === 1;
    });
    acc = '';
    let run = 0; // the echo never reaches past three words in a row that are not hers
    for (let i = 0; i < words.length; i++) {
      run = hers[i] ? 0 : run + 1;
      if (run >= 3) break;
      acc += norm(words[i]);
      if (i > end && i > 0 && hers[i] && hers[i - 1] && acc.length >= minLetters && found(grams(acc), h) >= ECHO) end = i;
    }
  }
  return end;
}

// userWords: the user's words of a transcript, past the echo (pastEcho's tokens).
export const userWords = (text, spoken) => toks(pastEcho(text, spoken));

// allHers: every word of a transcript is in one window, in the order said,
// each on its own letters (no joins): the growing edge of an interim, where
// the last word is still a near-miss ("manage to" heard as "managed to").
// A word sits where its first trigram found after the word before it is.
function allHers(text, spoken, minLetters = 8) {
  const ws = text.split(/\s+/).map(norm).filter(Boolean);
  if (ws.join('').length < minLetters) return false; // short phrases ("can we") are everyone's
  for (const sp of windows(spoken)) {
    const h = grams(sp);
    let pos = 0;
    const inOrder = ws.every(w => {
      if (w.length < 3) { const i = sp.indexOf(w, pos); return i >= 0 && (pos = i + 1); }
      const g = grams(w);
      if (found(g, h) < ECHO) return false;
      const at = [...g].map(x => sp.indexOf(x, pos)).filter(i => i >= 0);
      return at.length > 0 && (pos = Math.min(...at) + 1);
    });
    if (inOrder) return true;
  }
  return false;
}

// novelRun: two words in a row (of three letters or more; shorter ones are
// everyone's and skipped) that are in no window of what she said, each on
// its own letters — the user's speech, whatever her echo around it ("That
// sounds like a fun little experiment can we continue to explore how
// different": her words, the user's, hers again; Chrome, 2026-10-24)
function novelRun(text, spoken) {
  const hs = [...windows(spoken)].map(grams);
  let run = 0;
  for (const w of text.split(/\s+/).map(norm).filter(w => w.length >= 3)) {
    const g = grams(w);
    run = hs.some(h => found(g, h) >= ECHO) ? 0 : run + 1;
    if (run >= 2) return true;
  }
  return false;
}

// isEcho: a transcript that is all hers — nothing is left past its echo,
// or what is left still measures ECHO in a window of what she said ("Alf
// World" + "the house" over "ALFWorld: a household scene…"), or is her
// words, near-missed; and no run of the user's words (novelRun) inside it.
// A long echo never hides the user's words after it, or between its parts.
// A stop (keywordStop) is never echo: "stop" is heard at any time.
export function isEcho(text, spoken, stopWords = STOP_WORDS) {
  if (!toks(text).length || !units(spoken).length || keywordStop(text, stopWords)) return false;
  const rest = pastEcho(text, spoken);
  return rest === '' || (!novelRun(rest, spoken) && (echoShare(rest, spoken) >= ECHO || allHers(rest, spoken)));
}

export class Dialog {
  constructor({ fsttm, attention, tts, llm, bargeIn = 'words', stopWords = STOP_WORDS,
    log = () => {}, now = () => performance.now(), onTurn = () => {}, onHeard = () => {}, minWords = 2, autoResumeMs = 1500,
    echoMs = 12000, echoGapMs = 1000, textEcho = true }) {
    Object.assign(this, { fsttm, attention, tts, llm, bargeIn, stopWords, log, now, onTurn, onHeard, minWords, autoResumeMs, echoMs, echoGapMs, textEcho });
    this.seq = 0;             // turn ids: t1, t2, …
    this.speakingId = null;   // the turn being spoken (null: an announcement)
    // the narration a cut left unsaid: {units, entry} — entry is the history
    // turn holding what was heard; resuming speaks the units and grows it
    this.resume = null;
    this.resumeTimer = null;
    this.wordsSinceCut = false;
    this.speakingResume = null; // the entry a resume in progress grows
    this.history = [];       // {role: user|system, text, interrupted?}
    this.parked = [];        // system utterances waiting for a free floor
    this.abort = null;       // AbortController of the LLM call in flight
    this.onset = null;       // performance time of the user's onset
    this.cutInterim = null;  // the interim that triggered the last cut
    this.lastFinalAt = null;
    this.lastSpoken = []; // {sentences, until}: what she said in the last echoMs, for late echoes
    this.lastEnd = -Infinity; // when she last stopped speaking
    this.seg = null;         // the recognition segment in progress: {quiet}
    tts.onnarrated = text => this.remember(text); // the narrator's sentences (narrator.js) are echo too
    fsttm.onChange = (ev, src, dst) => {
      this.log('fsttm', { event: ev, src, dst });
      if ((dst === 'FREEu' || dst === 'FREEs') && this.parked.length) queueMicrotask(() => this.flush());
    };
  }

  get state() { return this.fsttm.state; }

  // spoken is what the recognizer may hear of us now, as sentences in the
  // order said: every utterance that ended in the last echoMs (a backlog of
  // narration lines is said in a row and its finals come late), then the one
  // being spoken, and the sentence the narrator reads (tts.narrating)
  spoken() {
    const t = this.now();
    this.lastSpoken = this.lastSpoken.filter(r => t < r.until);
    const cur = this.tts.speaking ? (this.tts.cur?.sentences ?? [this.tts.sentence]) : [];
    return [...this.lastSpoken.flatMap(r => r.sentences), ...cur, this.tts.narrating].filter(Boolean);
  }

  remember(text) {
    this.lastEnd = this.now();
    this.lastSpoken.push({ sentences: splitSentences(text), until: this.lastEnd + this.echoMs });
  }

  // heardOf is what of her a transcript of the segment in progress may hold.
  // A segment's first interim comes while she speaks, or within the
  // recognizer's lag (echoGapMs) after: then it may be echo. A segment that
  // began with her quiet longer than that holds none, so the user's words
  // are theirs even when they repeat hers ("put the apple in the fridge"
  // after the card line's "put the apple on the countertop"). A final with
  // no interim before it (a recognizer without them) is measured as before.
  // textEcho false (the recognizer listens through the echo canceller:
  // her voice never reaches it) measures nothing: the only words it could
  // drop are the user's ("how are you today" after her "…with you today").
  heardOf(first) {
    if (!this.textEcho) return [];
    if (first && !this.seg) this.seg = { quiet: !this.tts.speaking && !this.tts.narrating && this.now() - this.lastEnd > this.echoGapMs };
    return this.seg?.quiet ? [] : this.spoken();
  }

  // ── recognizer signals ────────────────────────────────────────────────
  speechStart() {
    this.onset = this.now();
    this.log('onset', { state: this.state, asleep: !this.attention.awake() });
    if (!this.attention.awake()) return; // ASLEEP: only a final's wake word counts
    if (this.fsttm.isSystem() && this.state === 'SYSTEM') {
      this.fsttm.tryAction('user', 'G'); // SYSTEM -(K,G)-> BOTHs: barge-in attempt
      if (this.bargeIn === 'energy') this.cut('energy');
      return;
    }
    if (this.state === 'FREEu' || this.state === 'FREEs') this.fsttm.tryAction('user', 'G'); // tentative grab
  }

  interim(text) {
    if (!text.trim()) return;
    const spoken = this.heardOf(true);
    if (isEcho(text, spoken, this.stopWords)) {
      this.log('echo', { text });
      return;
    }
    if (spoken.length) text = pastEcho(text, spoken); // the user's words past the echo
    if (!text.trim()) return;
    this.wordsSinceCut = true;
    // Chrome fires speechstart once per recognition session, so an onset
    // older than the last final is stale: the first user word of this turn is it
    if (this.onset == null) { this.onset = this.now(); this.log('onset', { state: this.state, from: 'interim' }); }
    this.log('interim', { text, state: this.state });
    if (!this.attention.awake()) return;
    const busy = this.tts.speaking || this.abort;
    if (busy && keywordStop(text, this.stopWords)) { // "stop" on a partial: at once, any policy
      this.cutInterim = text;
      this.cut('keyword');
      return;
    }
    // user words while she holds the floor are an onset even when Chrome sent
    // no speechstart (it sends one per recognition session): SYSTEM → BOTHs
    if (this.state === 'SYSTEM' && this.tts.speaking) {
      this.fsttm.tryAction('user', 'G');
      if (this.bargeIn === 'energy') { this.cut('energy'); return; }
    }
    // words: the overlap holds (BOTHs) until the user's words are sure — two
    // of them, or a final; one word over her speech may be a misrecognition
    if (this.state === 'BOTHs' && this.bargeIn === 'words' && toks(text).length >= this.minWords) {
      this.cutInterim = text; this.cut('words'); return;
    }
    if (this.state === 'FREEu' || this.state === 'FREEs') this.fsttm.tryAction('user', 'G'); // onset missed: the words confirm the grab
  }

  async final(text) {
    text = text.trim();
    this.log('final', { text, state: this.state, cutInterim: this.cutInterim });
    this.cutInterim = null;
    this.finalOnset = this.onset; // a cut at this final still measures from it
    this.onset = null; // the next turn's onset is its own
    const spoken = this.heardOf(false);
    this.seg = null; // the next interim starts a segment of its own
    if (!text) return;
    if (isEcho(text, spoken, this.stopWords)) { this.log('echo', { text, final: true }); return; }
    const raw = text;
    if (spoken.length) text = pastEcho(text, spoken);
    // heard: the user's words of the final, past her echo (a page shows these, not the raw final)
    this.log('heard', { text, echo: raw.slice(0, raw.length - text.length).trim(), state: this.state });
    const d = this.attention.onUtterance(text);
    this.log('attention', { action: d.action, state: this.attention.state });
    if (d.action === 'ignore') return;
    if (this.state === 'SYSTEM') this.fsttm.tryAction('user', 'G'); // words with no onset or interim before them
    if (this.state === 'BOTHs') this.cut(this.bargeIn === 'none' ? 'none' : 'final'); // policy none cuts only now
    if (this.state === 'FREEu' || this.state === 'FREEs') this.fsttm.tryAction('user', 'G');
    this.fsttm.tryAction('user', 'R'); // USER -(W,R)-> FREEu: the utterance goes to the LLM
    this.lastFinalAt = this.now();
    if (d.action === 'sleep') return this.say(['Voice controls disabled.'], 'sleep');
    const cmd = d.action === 'wake' ? d.text : text;
    if (!cmd) return; // the wake word alone: awake and listening
    this.wordsSinceCut = true;
    clearTimeout(this.resumeTimer);
    if (this.resume && resumePhrase(cmd)) { this.log('resume', { why: 'phrase', text: cmd }); this.resumeNow(); return; }
    this.history.push({ role: 'user', text: cmd });
    if (keywordStop(cmd, this.stopWords)) { this.log('keyword_stop', { text: cmd }); return; }
    await this.reply(cmd);
  }

  // ── the system side ───────────────────────────────────────────────────
  async reply(text) {
    this.resume = null; // a new turn: what was left unsaid stays unsaid
    this.abort?.abort();
    const ctl = new AbortController();
    this.abort = ctl;
    const t0 = this.now();
    const id = `t${++this.seq}`;
    let r;
    try {
      r = await this.llm(text, { signal: ctl.signal, history: this.history, id });
    } catch (e) {
      if (ctl.signal.aborted) { this.log('llm_aborted', { text }); return; }
      this.log('llm_error', { error: String(e) });
      return;
    } finally {
      if (this.abort === ctl) this.abort = null;
    }
    if (ctl.signal.aborted) return;
    this.log('llm', { text, ms: this.now() - t0, firstMs: r.firstMs, voice: r.voice, json: r.json, mode: r.mode });
    this.onTurn(r);
    if (r.voice) this.say(splitSentences(r.voice), 'reply', id);
  }

  // say takes the floor when it is free (FREEu/FREEs) and speaks; while the
  // user holds it, the utterance waits (W) — never talk over the user.
  say(sentences, why = 'say', id = null) {
    if (!this.attention.awake() && why !== 'sleep') { this.log('parked', { why, asleep: true }); this.parked.push({ sentences, why, id }); return; }
    if (this.state === 'USER' || this.state === 'BOTHu' || this.tts.speaking) {
      this.log('parked', { why, state: this.state });
      this.parked.push({ sentences, why, id });
      return;
    }
    if (!this.fsttm.tryAction('system', 'G')) { this.parked.push({ sentences, why, id }); return; }
    const entry = why === 'resume' ? this.speakingResume : null;
    this.speakingId = id;
    const t = this.now();
    this.log('speak', { why, sentences, freeuDwell: this.lastFinalAt != null ? t - this.lastFinalAt : null });
    this.lastFinalAt = null;
    this.tts.speak(sentences, {
      onstart: i => this.log('tts_start', { i }),
      onend: completed => {
        if (!completed) return; // a cut releases the floor itself
        this.remember(sentences.join(' '));
        if (entry) {
          entry.text = `${entry.text} ${sentences.join(' ')}`.trim(); entry.interrupted = false; this.speakingResume = null;
          if (entry.id) this.onHeard(entry.id, entry.text);
        } else {
          this.history.push({ role: 'system', text: sentences.join(' '), id });
          if (id) this.onHeard(id, sentences.join(' ')); // heard whole
        }
        this.fsttm.tryAction('system', 'R'); // SYSTEM -(R,W)-> FREEs on audio end, never on text end
        this.log('tts_end', { completed });
      },
    });
  }

  // announce is system-initiated: on a free floor at once; over the user
  // only when STOP-class (system G → BOTHu), else parked.
  announce(text, { stopClass = false } = {}) {
    if (stopClass && this.state === 'USER') {
      this.fsttm.tryAction('system', 'G'); // USER -(G,K)-> BOTHu
      this.log('announce', { text, over: 'user' });
      this.tts.speak([text], { onend: c => { if (c) { this.remember(text); this.fsttm.tryAction('system', 'R'); } } });
      return;
    }
    this.say([text], 'announce');
  }

  flush() {
    if (!this.parked.length || this.state === 'USER' || this.tts.speaking || !this.attention.awake()) return;
    const p = this.parked.shift();
    this.say(p.sentences, p.why, p.id);
  }

  // cut is the system's R during overlap: TTS cancelled, the LLM stream
  // aborted, the turn recorded as heard and marked [interrupted].
  cut(why) {
    const cur = this.tts.cur;
    const all = cur?.sentences?.join(' ') ?? '';
    const heard = this.tts.cancel();
    if (all) this.remember(all);
    // the context cut: the history keeps only what was heard, as if the rest
    // was never generated (the next prompt shares the KV prefix up to it); the
    // rest — the tail of the sentence being spoken, then the sentences after
    // it — waits for "continue" or an auto-resume
    let rest = [];
    if (cur && heard !== null) {
      const done = cur.sentences.slice(0, cur.i).join(' ');
      const part = heard.slice(done.length).trim();
      const s = cur.sentences[cur.i] ?? '';
      rest = [s.startsWith(part) ? s.slice(part.length).trim() : s, ...cur.sentences.slice(cur.i + 1)].filter(Boolean);
    }
    if (this.abort) { this.abort.abort(); this.abort = null; }
    const onset = this.onset ?? this.finalOnset;
    this.finalOnset = null;
    const ms = onset != null ? this.now() - onset : null;
    const estimated = !!this.tts.estimated;
    let entry = null;
    if (heard !== null) {
      entry = this.speakingResume; // a resume cut again: its turn grows
      if (entry) { entry.text = `${entry.text} ${heard}`.trim(); entry.interrupted = true; }
      else { entry = { role: 'system', text: heard, interrupted: true, estimated, id: this.speakingId }; this.history.push(entry); }
      this.speakingResume = null;
      if (entry.id) this.onHeard(entry.id, entry.text);
    }
    this.resume = rest.length && entry ? { units: rest, entry } : null;
    this.wordsSinceCut = why === 'words' || why === 'final' || why === 'none';
    clearTimeout(this.resumeTimer);
    // a cut no words confirmed (energy) was maybe a cough: with no words
    // within autoResumeMs, she goes on (the Python server's auto-resume)
    if (this.resume && !this.wordsSinceCut) this.resumeTimer = setTimeout(() => {
      if (!this.wordsSinceCut && this.resume) { this.log('resume', { why: 'auto (no words after the cut)' }); this.resumeNow(); }
    }, this.autoResumeMs);
    if (this.state === 'BOTHs' || this.state === 'SYSTEM' || this.state === 'BOTHu') this.fsttm.tryAction('system', 'R');
    this.log('cut', { why, onsetToCutMs: ms, heard, estimated, interim: this.cutInterim, state: this.state, rest: rest.join(' ') });
  }

  // resumeNow speaks what the cut left unsaid, from the word it stopped at
  resumeNow() {
    const r = this.resume;
    this.resume = null;
    if (!r) return;
    if (this.state === 'USER' || this.state === 'BOTHu') this.fsttm.tryAction('user', 'R'); // the user let go
    this.speakingResume = r.entry;
    this.say(r.units, 'resume', r.entry.id);
  }
}

// cancelledMic: the mic with the browser's echo canceller (and noise
// suppression), for the recognizer to listen to instead of the raw mic: her
// voice never reaches it. Measured with web/mic.html (Chrome, Linux, laptop
// speakers, 2026-10-24): the canceller takes ~40 dB off her speechSynthesis
// voice, below the room's quiet; the recognizer on this track heard none of
// her sentence and all of the user's.
export const cancelledMic = () => navigator.mediaDevices.getUserMedia({
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false } });

// attachRecognizer wires a (webkit)SpeechRecognition to the dialog and
// keeps it running (Chrome ends a continuous session after silence). With
// a track (cancelledMic's), the recognizer listens to it (start(track),
// newer Chrome); a browser that refuses it listens to the raw mic, and the
// textual echo rejection is the only one (logged rec_track).
export function attachRecognizer(rec, dialog, { lang = 'en-US', track = null } = {}) {
  rec.lang = lang;
  rec.continuous = true;
  rec.interimResults = true;
  let want = true, onTrack = !!track;
  const start = () => {
    if (onTrack) try { rec.start(track); return; } catch (e) {
      onTrack = false; dialog.textEcho = true; dialog.log('rec_track', { used: false, error: `${e.name}: ${e.message}` });
    }
    rec.start();
  };
  rec.onspeechstart = () => dialog.speechStart();
  rec.onresult = e => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) dialog.final(r[0].transcript); else dialog.interim(r[0].transcript);
    }
  };
  rec.onerror = e => dialog.log('rec_error', { error: e.error, message: e.message });
  rec.onend = () => { dialog.log('rec_end', {}); if (want) try { start(); } catch {} };
  start();
  if (onTrack) dialog.log('rec_track', { used: true, label: track.label });
  dialog.textEcho = !onTrack; // through the canceller her voice never reaches it: no text to measure

  return { stop() { want = false; rec.stop(); }, get track() { return onTrack ? track : null; } };
}
