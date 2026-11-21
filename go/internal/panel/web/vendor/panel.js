// panel.js — the dialog as a widget panel: the one voice/text input (the mic, the text
// line, the floor and the input's mode) and the intent chat (what was heard or typed,
// the intent the backend made of it, the backend's words), over the library's floor
// (dialog.js) and its wire (ws.js, or any adapter with the same turn/cut/heard).
//
//   const panel = new Panel(document.getElementById('dock'), { backend, table });
//   window.say = text => panel.say(text);       // the panel's public entry
//
// One entry for every input. A typed line, a recognized utterance and panel.say(text)
// (a script, a test, a backend's own text socket) all take the same path: the floor's
// final(), then a turn on the backend; the mode and the floor rules apply to each alike.
//
// One channel: the backend (voice-nav-removal, the user 2026-10-24: one voice paradigm, the
// dialog with the agent and the game — the second channel, its flip and its push-to-talk
// quasimode, served only the voice navigation and are gone). The page's own state of the
// input (a mode its rules keep, e.g. dictation) is shown with setMode(text).
//
// The controls the dialog has, in the panel's bar (the strip): the barge-in policy
// (words / energy / none, live on the dialog), attention (awake / asleep, where there is a
// wake word), the reply mode (where the backend has modes: `modes`), beside the mic and the
// voice switch of the input line. The dialog's insides, as foldable sections a page shows
// or not (`sections: ['floor', 'state', 'turns', 'events']`, closed until opened, the open
// ones kept per page): the floor machine (fsmview.js), the state and the numbers (bench.js),
// the turns' timings, the event log (saved as JSONL).
//
// The host's backchannel (the README's "The wire"), on the backend's socket: the host types as
// the person (chat{who:"user"}, the one entry), speaks as her (chat{who:"system"}) and sets the
// strip (ctl{…}); a page that opts in (`host: true`, or {rows, events, state}) sends back every
// row as it is tagged (chat{…}), every dialog event (event{…}) and the state on each change
// (state{…}) — a test or a program on the host reads the dialog as the person sees it.
//
// PanelCore is the logic (no DOM: node tests drive it, test/panel.mjs); Panel renders
// it into an element the page gives (a dock, a sidebar, a whole page), with lib/panel.css.

import { FSTTM } from './fsttm.js';
import { Attention } from './attention.js';
import { Dialog, attachRecognizer, cancelledMic } from './dialog.js';
import { BrowserTTS, splitSentences } from './tts.js';
import { Bench } from './bench.js';
import { FsmView } from './fsmview.js';

export const BARGE_IN = ['words', 'energy', 'none'];

export class PanelCore {
  // backend: turn(id, text, mode, {signal}) → sentences, cut(id, heard), heard(id, text),
  //   info[id].json (the intent's JSON, when the backend has one) and an onFrame slot;
  // table: lib/fsttm.json; tts: BrowserTTS-like
  // modes: the backend's reply modes, each a value or {value, label} (the first is the
  //   default, or reply); turn(id, text, mode) gets the one chosen ('intent' without modes)
  // narrate(frame): is a narration voiced (default: unless form R); backendName: shown
  // host: the backchannel to the host — true, or {rows, events, state}: what goes back (backend.send)
  // hostLink: the backchannel on a socket of its own ({send, onFrame, onOpen?}: hostSocket(url), the
  //   fsttm-chat tool's /page), where the backend has none (llama-server) or the page is driven from
  //   a terminal; host defaults to true with it
  // frames: the application's own frames kept as events (kind frame, its ev and kind as ev / of):
  //   false (default: a busy backend would fill the log), true, or frame => bool
  // onRow(row), onState(state): the view's; onFrame(frame): a frame the panel does not
  // know (the application's own); log(kind, fields): every dialog event; onEvent(event):
  // the same as the bench keeps it ({t, kind, …}), for a view
  constructor({ backend, table, tts = null, attention = null, bargeIn = 'words', voice = true,
    modes = null, reply = null, narrate = null, backendName = null, bench = null, host = null, hostLink = null, frames = false,
    stopWords, onRow = () => {}, onState = () => {}, onFrame = () => {}, log = () => {},
    onEvent = () => {}, now, dialog = null, maxRows = 300 }) {
    Object.assign(this, { backend, voice, onRow, onState, onFrame, log, onEvent, maxRows, backendName });
    this.modes = modes?.map(m => typeof m === 'string' ? { value: m, label: m } : m) ?? null;
    this.reply = reply ?? this.modes?.[0]?.value ?? 'intent';
    this.narrate = narrate ?? (f => f.form !== 'R');
    this.bench = bench ?? new Bench(now ? { now } : {});
    this.timings = [];           // the turns' timings: {n, mode, text, first, ms, dwell, end}
    this.recognition = null;     // what the recognizer probe said (probeLocal)
    this.hostLink = hostLink;
    host ??= !!hostLink;
    this.host = host === true ? { rows: true, events: true, state: true } : host || null;
    this.hostN = 0;              // the host's own system lines: h1, h2, …
    this.keepFrame = typeof frames === 'function' ? frames : () => !!frames;
    this.onMic = null;           // ctl{mic}: the view's (Panel opens the mic)
    this.rows = [];
    this.floor = 'FREEs';
    this.mode = '';              // the page's word for its input's mode
    this.listening = false;
    this.via = null;             // the source of the final in progress: text | voice | say (the dialog calls turn() inside final(), synchronously)
    this.turns = new Map();      // dialog turn id → { person, agent } rows
    this.tts = tts ?? new BrowserTTS();
    this.dialog = dialog ?? new Dialog({
      fsttm: new FSTTM(table, now ? { now } : {}), attention: attention ?? new Attention({ enabled: false, startAsleep: false }),
      tts: this.tts, bargeIn, ...(stopWords ? { stopWords } : {}), ...(now ? { now } : {}),
      log: (k, f) => this.event(k, f),
      llm: (text, o) => this.turn(text, o),
      onHeard: (id, heard) => this.heard(id, heard),
    });
    if (backend && 'onFrame' in backend) backend.onFrame = f => this.frame(f);
    if (hostLink) {
      hostLink.onFrame = f => this.frame(f);
      if ('onOpen' in hostLink) hostLink.onOpen = () => this.hostOpened();
    }
  }

  state() {
    const a = this.dialog.attention;
    return { floor: this.floor, mode: this.mode, listening: this.listening,
      voice: this.voice, bargeIn: this.dialog.bargeIn,
      attention: a.awake() ? 'AWAKE' : 'ASLEEP', wakeable: !!a.enabled, reply: this.reply, modes: this.modes,
      recognizer: this.recognizerName() };
  }
  // the recognizer: off, or what it listens to (the echo-cancelled track or the raw mic) and where it runs
  recognizerName() {
    if (!this.listening) return 'off';
    const where = this.recognition?.processLocally ? ' · on-device' : this.recognition?.api ? ' · cloud' : '';
    return `${this.recognizer?.track ? 'echo-cancelled track' : 'raw mic'}${where}`;
  }
  changed() {
    const s = this.state();
    this.onState(s);
    if (this.host?.state) {      // the state on the wire: on each change, not on each call
      const w = { ev: 'state', floor: s.floor, attention: s.attention, mode: s.mode, reply: s.reply,
        barge_in: s.bargeIn, recognizer: s.recognizer, voice: s.voice, listening: s.listening };
      const j = JSON.stringify(w);
      if (j !== this.hostState) { this.hostState = j; this.toHost(w); }
    }
  }
  // toHost: a frame back to the host on the link, else the backend's socket (a backend without send, or a closed one: dropped)
  toHost(frame) { try { (this.hostLink ?? this.backend)?.send?.(frame); } catch {} }
  // the host's socket is open (again): it gets the state now, not at the next change
  hostOpened() { this.hostState = null; this.changed(); }
  // a row as the host reads it: who (user, system, intent, note), its words, its source, its tags
  hostRow(r) {
    const who = { person: 'user', agent: 'system', intent: 'intent', note: 'note' }[r.who] ?? r.who;
    const via = r.who === 'person' ? ({ text: 'typed', voice: 'heard' }[r.via] ?? r.via) : r.via ?? (r.who === 'note' ? 'panel' : 'backend');
    this.toHost({ ev: 'chat', who, text: r.text, row: r.n, id: r.turn ?? null, via, tags: [...r.tags],
      ...(r.narration ? { kind: r.narration === true ? null : r.narration } : {}),
      ...(r.hostId != null ? { host_id: r.hostId } : {}) });
  }

  row(r) {
    const row = { n: (this.rows.at(-1)?.n ?? 0) + 1, tags: [], ...r };
    this.rows.push(row);
    if (this.rows.length > this.maxRows) this.rows.splice(0, this.rows.length - this.maxRows);
    while (this.turns.size > this.maxRows) this.turns.delete(this.turns.keys().next().value);
    this.onRow(row, 'add');
    if (this.host?.rows) this.hostRow(row);
    return row;
  }
  tag(row, t) { if (row) { row.tags.push(t); this.onRow(row, 'tag'); if (this.host?.rows) this.hostRow(row); } }

  // ── the input ───────────────────────────────────────────────────────────
  // say: the public entry (a typed line, a script, a backend's text socket)
  say(text, { via = 'say' } = {}) {
    text = String(text ?? '').trim();
    if (!text) return Promise.resolve();
    if (via === 'text') {          // typing is addressing her: no wake word needed
      if (this.dialog.attention.force(true)) this.event('attention', { state: 'AWAKE', forced: 'typed' });
      this.event('typed', { text });
    }
    this.via = via;                // an utterance the floor drops (a stop phrase, an echo) never reaches turn(): nothing to undo
    try { return this.dialog.final(text); } finally { this.via = null; }
  }
  // listen: the mic, a SpeechRecognition (new SR()); what it finalizes goes down say().
  // opts.stream: the echo-cancelled mic (cancelledMic()) the recognizer listens to — her
  // voice never reaches it, and the dialog's textual echo rejection is off; unlisten stops it
  listen(rec, { stream = null, ...opts } = {}) {
    if (this.recognizer) return;
    const core = this, dialog = this.dialog;
    this.stream = stream;
    this.recognizer = attachRecognizer(rec, {
      speechStart: () => dialog.speechStart(),
      interim: t => dialog.interim(t),
      final: t => core.say(t, { via: 'voice' }),
      log: (k, f) => dialog.log(k, f),
      set textEcho(v) { dialog.textEcho = v; },
      get textEcho() { return dialog.textEcho; },
    }, { ...opts, track: stream?.getAudioTracks()[0] ?? null });
    this.listening = true;
    this.changed();
  }
  unlisten() {
    this.recognizer?.stop();
    this.recognizer = null;
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
    this.listening = false;
    this.changed();
  }
  setVoice(on) { this.voice = !!on; if (!on) this.tts.cancel?.(); this.changed(); }
  setMode(mode) { this.mode = mode ?? ''; this.changed(); }
  // the strip's controls: the barge-in policy (live on the dialog), attention, the reply mode
  setBargeIn(policy) {
    if (!BARGE_IN.includes(policy)) return false;
    this.dialog.bargeIn = policy;
    this.event('policy', { bargeIn: policy });
    this.changed();
    return true;
  }
  setAttention(awake, forced = 'switch') {
    const a = this.dialog.attention;
    if (a.force(!!awake)) this.event('attention', { state: a.state, forced });
    this.changed();
  }
  setReply(value) {
    if (this.modes && !this.modes.some(m => m.value === value)) return false;
    this.reply = value;
    this.changed();
    return true;
  }

  // ── a turn ───────────────────────────────────────────────────────────────
  // the dialog's llm: the utterance went past the floor, the echo and attention rules
  async turn(text, { id, signal } = {}) {
    const via = this.via ?? 'say';
    const mode = this.reply;
    const echo = this.echoBefore; this.echoBefore = null;
    const hostId = via === 'host' ? this.hostId : null; this.hostId = null;
    const person = this.row({ who: 'person', text, via, turn: id, meta: `${via} · ${this.floor}`, ...(hostId != null ? { hostId } : {}),
      tags: echo ? [`her echo before it dropped: "${echo}"`] : [] });
    const t0 = performance.now();
    let words = '', agent = null;
    try {
      const sentences = [];
      for await (const s of this.backend.turn(id, text, mode, { signal })) sentences.push(s);
      const json = this.backend.info?.[id]?.json;
      if (json) this.row({ who: 'intent', text: json, turn: id });
      words = sentences.join(' ');
      if (words) agent = this.row({ who: 'agent', text: words, turn: id });
    } catch (e) {
      if (signal?.aborted) throw e;
      this.row({ who: 'note', text: `the backend: ${e.message ?? e}`, turn: id });
      throw e;
    }
    this.turns.set(id, { person, agent });
    return { voice: this.voice ? words : '', firstMs: this.backend?.info?.[id]?.firstMs ?? performance.now() - t0,
      mode: this.modes?.find(m => m.value === mode)?.label ?? mode };
  }

  // ── the backend's own frames ─────────────────────────────────────────────
  frame(f) {
    if (f.ev === 'chat') return this.hostChat(f);
    if (f.ev === 'ctl') return this.hostCtl(f);
    if (f.ev === 'narration') {
      // a v3-turn narration is also the turn's voice (metis-agent WIRE.md: "do not voice it
      // twice"): the turn's own row says it; as a row of its own it was shown and spoken twice
      if (f.kind === 'v3-turn') return;
      const text = f.form === 'R' ? (f.line || f.sentence || '') : (f.sentence || f.line || '');
      if (!text) return;
      const voiced = f.via === 'host' ? f.form !== 'R' : this.narrate(f);   // the host's system words: voiced unless voice false
      this.event('narration', { id: f.id, of: f.kind, form: f.form, path: f.path, voiced });
      const row = this.row({ who: 'agent', text, narration: f.kind ?? true, form: f.form, ...(f.via ? { via: f.via } : {}),
        ...(f.hostId != null ? { hostId: f.hostId } : {}),
        meta: `${f.kind ?? 'narration'} · ${f.form ?? 'N'}${f.path ? ' · ' + f.path : ''}${voiced ? '' : ' · shown, not voiced'}` });
      const key = `n:${f.id}`;
      this.turns.set(key, { agent: row });
      if (this.voice && voiced) this.dialog.say(splitSentences(text), 'narration', key);
      return;
    }
    // the application's own frame: the page's (onFrame); an event of kind frame only where the page keeps them
    // (its ev names are not the dialog's kinds: a "cut" of the application's is no barge-in)
    if (this.keepFrame(f)) { const { t, kind, ...rest } = f; this.event('frame', kind !== undefined ? { ...rest, of: kind } : rest); }
    this.onFrame(f);
  }

  // the host's chat: as the person (the one entry: the floor, attention, a turn), or as her (a narration, announce)
  hostChat(f) {
    const text = String(f.text ?? '').trim();
    if (!text) return;
    this.event('host', { chat: f.who, text });
    if (f.who === 'user') { this.hostId = f.id ?? null; return this.say(text, { via: 'host' }); }   // its id comes back on the row (host_id)
    if (f.who === 'system') return this.frame({ ev: 'narration', id: f.id ?? `h${++this.hostN}`, kind: 'announce',
      form: f.voice === false ? 'R' : 'N', sentence: text, line: text, via: 'host', hostId: f.id ?? null });
  }
  // the host's ctl: the strip's controls (mode is the reply mode's other name; mic needs the view)
  hostCtl(f) {
    const { ev, ...c } = f;
    this.event('ctl', c);
    if (c.barge_in != null) this.setBargeIn(c.barge_in);
    if (c.voice != null) this.setVoice(!!c.voice);
    if (c.attention != null) this.setAttention(c.attention === true || /^awake$/i.test(String(c.attention)), 'host');
    if ((c.reply ?? c.mode) != null) this.setReply(c.reply ?? c.mode);
    if (c.mic != null) this.onMic?.(!!c.mic);
  }

  heard(id, heard) {
    if (id?.startsWith('n:')) this.backend?.heard?.(id.slice(2), heard);
    else if (id) this.backend?.cut?.(id, heard);
  }

  // the dialog's events: the floor, what was spoken, a cut; each kept (bench) and timed
  event(kind, f = {}) {
    const e = this.bench.log(kind, f);
    if (kind === 'fsttm') { this.floor = f.dst; this.changed(); }
    else if (kind === 'speak') {
      this.speaking = this.turns.get(this.dialog.speakingId)?.agent ?? null;
      this.tag(this.speaking, 'spoken');
    } else if (kind === 'cut') {
      this.tag(this.speaking, `cut · heard "${f.heard ?? ''}"${f.estimated ? ' (estimated)' : ''}`);
    } else if (kind === 'heard') this.echoBefore = f.echo || null;            // the final past her echo: turn() tags its row
    else if (kind === 'recognition') this.recognition = f;
    else if (kind === 'attention') {
      if (f.action === 'ignore') this.row({ who: 'note', text: `ignored: asleep — say "${this.dialog.attention.wake?.at(-1) ?? ''} …", wake it, or type` });
      this.changed();
    } else if (kind === 'keyword_stop') this.row({ who: 'note', text: `stop: "${f.text}"` });
    else if (kind === 'resume') this.row({ who: 'note', text: `resuming (${f.why}) from the cut word` });
    else if (kind === 'rec_error' && f.error !== 'no-speech' && f.error !== 'aborted') this.row({ who: 'note', text: `the microphone: ${f.error} ${f.message ?? ''}`.trim() });
    this.timing(e);
    this.log(kind, f);        // the page's: keyword_stop (a stop phrase never reaches turn()), echo, …
    this.onEvent(e);
    if (this.host?.events && kind !== 'frame') this.toHost({ ...e, ev: 'event' });   // an application's frame never goes back: the host would get its own
  }

  // the turns' timings (the turns section): final → first token, the reply, the dwell, how it ended
  timing(e) {
    const cur = this.timings.at(-1), ms = v => v != null ? `${Math.round(v)} ms` : '—';
    if (e.kind === 'llm') this.timings.push({ n: this.timings.length + 1, mode: e.mode ?? '?', text: e.text, first: e.firstMs, ms: e.ms, dwell: null, end: 'parked' });
    else if (e.kind === 'llm_aborted') this.timings.push({ n: this.timings.length + 1, mode: this.reply, text: e.text, end: 'aborted (new words)' });
    else if (cur && e.kind === 'speak' && e.why === 'reply') { cur.dwell = e.freeuDwell; cur.end = 'speaking…'; }
    else if (cur && e.kind === 'speak' && e.why === 'resume') { cur.end = 'speaking…'; cur.resumed = (cur.resumed ?? 0) + 1; }
    else if (cur && e.kind === 'tts_end' && cur.end === 'speaking…') cur.end = cur.resumed ? `spoken · resumed ×${cur.resumed}` : 'spoken';
    else if (cur && e.kind === 'cut' && cur.end === 'speaking…') cur.end = `cut · ${e.why} · ${ms(e.onsetToCutMs)}`;
    else return;
    if (this.timings.length > this.maxRows) this.timings.splice(0, this.timings.length - this.maxRows);
  }
}

// ── the view ──────────────────────────────────────────────────────────────
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e[k] = v; else if (v != null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
};

// probeLocal: on-device recognition for en-US — what this browser says, and install()
// when it says downloadable (the language pack; Chrome)
export async function probeLocal(SR, lang = 'en-US') {
  const out = { api: !!SR, available: typeof SR?.available === 'function' };
  if (!out.available) return out;
  try {
    out.local = await SR.available({ langs: [lang], processLocally: true });
    out.cloud = await SR.available({ langs: [lang], processLocally: false });
    if (out.local === 'downloadable' && typeof SR.install === 'function') {
      out.install = await SR.install({ langs: [lang], processLocally: true });
      out.local = await SR.available({ langs: [lang], processLocally: true });
    }
    out.processLocally = out.local === 'available';
  } catch (e) { out.error = String(e); }
  return out;
}

// hostSocket: the backchannel on a socket of its own, opened again when it closes (the fsttm-chat
// tool, python -m fsttm_web.chat, at /page; or any Relay's page_route): a hostLink
export function hostSocket(url, { reconnect = 2000, WebSocketImpl = globalThis.WebSocket } = {}) {
  const link = {
    url, ws: null, closed: false, onFrame: null, onOpen: null,
    get isOpen() { return link.ws?.readyState === 1; },
    send(f) { if (!link.isOpen) throw new Error('host socket closed'); link.ws.send(JSON.stringify(f)); },
    close() { link.closed = true; link.ws?.close(); },
  };
  const open = () => {
    const ws = link.ws = new WebSocketImpl(url);
    ws.onopen = () => link.onOpen?.();
    ws.onmessage = e => { let f; try { f = JSON.parse(e.data); } catch { return; } link.onFrame?.(f); };
    ws.onclose = () => { if (!link.closed && reconnect > 0) setTimeout(open, reconnect); };
  };
  open();
  return link;
}
// the page's URL asks for the tool: ?fw_host (port 8395), ?fw_host=8396, ?fw_host=ws://…/page
export function hostURL(param = 'fw_host', search = globalThis.location?.search ?? '') {
  const v = param ? new URLSearchParams(search).get(param) : null;
  if (v == null) return null;
  return /^wss?:\/\//.test(v) ? v : `ws://127.0.0.1:${/^\d+$/.test(v) ? v : 8395}/page`;
}

export const SECTIONS = { floor: 'floor · FSTTM', state: 'state · perf', turns: 'turns · timing', events: 'events' };
const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} } };

export class Panel {
  // root: the element the panel fills; opts: PanelCore's, plus
  //   Recognizer (window.SpeechRecognition ?? webkitSpeechRecognition), title,
  //   mic (cancelledMic: the echo-cancelled mic the recognizer listens to; null: the raw
  //   mic, her voice dropped by text), onDevice (probe and ask for on-device recognition),
  //   micWakes (the mic button wakes her and its off puts her to sleep, where there is a
  //   wake word; default true),
  //   controls (the strip: true), sections (which of SECTIONS, in that order; none by
  //   default), storeKey (whose open sections are kept: the page's path), meta (rows show
  //   their small print: via and floor, a narration's kind and form);
  //   dialogIn (an element: the dialog's own part there — the floor, barge-in, attention and the
  //   sections as cards — a tools drawer the page may hide, showDialog(shown); the chat keeps the
  //   reply mode), hostParam ('fw_host': ?fw_host on the page's URL connects the backchannel to the
  //   fsttm-chat tool, hostURL(); null: never), micButton (the mic by the text line: true; false where the page has its own, micSwitch),
  //   voiceBox (a 'voice' tick by the text line: false; the voice is the header's switch, voiceWidget / voiceSwitch — the user, 2026-11-07),
  //   the page's onRow / onState / onEvent are called after the view has drawn; subscribe(fn)
  //   adds a listener later (a header's widget): fn({state} | {event})
  constructor(root, opts = {}) {
    const { title = 'dialog',
      Recognizer = globalThis.SpeechRecognition ?? globalThis.webkitSpeechRecognition, mic = cancelledMic,
      onDevice = false, micWakes = true, controls = true, sections = [], dialogIn = null, hostParam = 'fw_host', micButton = true, voiceBox = false,
      storeKey = globalThis.location?.pathname ?? '',
      meta = false, onRow = () => {}, onState = () => {}, onEvent = () => {}, ...core } = opts;
    Object.assign(this, { root, Recognizer, mic, onDevice, micWakes, meta });
    this.rowEl = new Map();
    this.subs = new Set();
    this.dialogShown = true;              // the dialog's part is on screen (dialogIn: the page says, showDialog)
    const tool = !core.hostLink && hostURL(hostParam);
    if (tool) core.hostLink = hostSocket(tool);
    this.core = new PanelCore({ ...core,                       // the page's onRow / onState after the view's
      onRow: (r, how) => { this.drawRow(r, how); onRow(r, how); }, onState: s => { this.drawState(s); onState(s); this.tell({ state: s }); },
      onEvent: e => { this.drawEvent(e); onEvent(e); this.tell({ event: e }); } });
    this.core.onMic = on => { if (on !== this.core.listening) this.toggleMic(); };   // the host's ctl{mic}
    root.classList.add('fw-panel');
    this.dom = {
      floor: el('span', { class: 'fw-floor' }), mode: el('span', { class: 'fw-mode' }),
      log: el('ol', { class: 'fw-log', 'aria-live': 'polite', 'aria-label': 'the dialog' }),
      text: el('input', { class: 'fw-text', type: 'text', autocomplete: 'off', placeholder: 'say or type', 'aria-label': 'say or type' }),
      mic: el('button', { class: 'fw-mic', type: 'button', 'aria-pressed': 'false' }, 'mic'),
      voice: el('input', { type: 'checkbox', checked: this.core.voice }),
    };
    const d = this.dom;
    const form = el('form', { class: 'fw-input' }, micButton ? d.mic : null, d.text, voiceBox ? el('label', { class: 'fw-voice' }, d.voice, ' voice') : null);
    const bar = el('div', { class: 'fw-bar' }, el('span', { class: 'fw-title' }, title), d.mode);
    this.sections = this.makeSections(sections.filter(n => n in SECTIONS), `fw-panel:${storeKey}:sections`);
    const box = this.sections.box ? [this.sections.box] : [];
    if (dialogIn) {                       // the chat here, the dialog's own part in the page's drawer
      const dbar = el('div', { class: 'fw-bar' }, d.floor);
      if (controls) { const s = this.strip(); dbar.append(s.dialog); bar.append(s.chat); }
      dialogIn.classList.add('fw-panel', 'fw-part');
      dialogIn.replaceChildren(dbar, ...box);
      this.dialogEl = dialogIn;
      root.replaceChildren(bar, el('div', { class: 'fw-main' }, d.log, form));
    } else {
      bar.append(d.floor);
      if (controls) { this.strip(); bar.append(el('span', { class: 'fw-controls' }, d.barge, d.attn, d.reply)); }
      root.replaceChildren(bar, el('div', { class: 'fw-main' }, d.log, form), ...box);
    }
    form.onsubmit = e => { e.preventDefault(); const t = d.text.value; d.text.value = ''; this.core.say(t, { via: 'text' }); };
    d.mic.onclick = () => this.toggleMic();
    d.voice.onchange = () => this.core.setVoice(d.voice.checked);
    if (!this.Recognizer) { d.mic.disabled = true; d.mic.title = 'no SpeechRecognition in this browser'; }
    this.drawState(this.core.state());
    if (tool) this.note(`the host's tool: ${tool} (fsttm-chat)`);
  }

  say(text) { return this.core.say(text); }
  subscribe(fn) { this.subs.add(fn); fn({ state: this.core.state() }); return () => this.subs.delete(fn); }
  tell(m) { for (const fn of this.subs) { try { fn(m); } catch (e) { console.error(e); } } }
  setMode(m) { this.core.setMode(m); }
  setBargeIn(p) { return this.core.setBargeIn(p); }
  note(text) { return this.core.row({ who: 'note', text }); }
  async toggleMic() {
    const core = this.core, wake = this.micWakes && core.dialog.attention.enabled;
    if (core.listening) { core.unlisten(); if (wake) core.setAttention(false, 'mic'); return; }
    if (!this.Recognizer || this.opening) return;
    this.opening = true;
    this.tell({ state: core.state() });                                     // a widget shows the mic opening
    try {
      if (this.onDevice && !core.recognition) core.event('recognition', await probeLocal(this.Recognizer));
      const stream = this.mic ? await this.mic().catch(() => null) : null; // refused or absent: the raw mic
      const rec = new this.Recognizer();
      if (core.recognition?.processLocally) rec.processLocally = true;
      core.listen(rec, { stream });
      if (wake) core.setAttention(true, 'mic');                             // pressing mic is mic possession
      if (!core.recognizer?.track) this.note('the mic without the echo canceller: her voice reaches the recognizer, dropped by text');
    } finally { this.opening = false; this.tell({ state: core.state() }); }
  }

  // the strip: barge-in, attention (where there is a wake word), the reply mode (where the backend has modes)
  strip() {
    const c = this.core, d = this.dom;
    d.barge = el('select', { class: 'fw-ctl', 'aria-label': 'barge-in', title: 'barge-in: what of the user cuts her — two words, any voice, or nothing until a final' },
      BARGE_IN.map(p => el('option', { value: p }, `barge-in: ${p}`)));
    d.barge.onchange = () => c.setBargeIn(d.barge.value);
    d.attn = el('button', { class: 'fw-ctl fw-attn', type: 'button', 'aria-pressed': 'false', title: 'attention: awake listens to everything, asleep only to the wake word' }, 'awake');
    d.attn.onclick = () => c.setAttention(!c.dialog.attention.awake());
    d.reply = el('select', { class: 'fw-ctl', 'aria-label': 'reply mode', title: "the backend's reply mode" },
      (c.modes ?? []).map(m => el('option', { value: m.value }, m.label)));
    d.reply.onchange = () => c.setReply(d.reply.value);
    return { dialog: el('span', { class: 'fw-controls' }, d.barge, d.attn), chat: el('span', { class: 'fw-controls' }, d.reply) };
  }

  // the sections: one <details> each, closed until opened; the open ones kept (storeKey)
  makeSections(names, key) {
    if (!names.length) return {};
    const open = store.get(key) ?? {}, sec = { names, key, open, el: {} };
    sec.box = el('div', { class: 'fw-sections' }, names.map(n => {
      const det = el('details', { class: `fw-sec fw-sec-${n}`, 'data-sec': n }, el('summary', {}, SECTIONS[n]), this.sectionBody(n));
      det.open = !!open[n];
      det.ontoggle = () => { sec.open[n] = det.open; store.set(key, sec.open); this.drawSection(n); this.ticking(); };
      sec.el[n] = det;
      return det;
    }));
    this.trail = [];                      // the last floor states, {state, t}
    this.evFrom = 0;                      // the events section shows the bench's events from here (clear)
    queueMicrotask(() => { for (const n of names) this.drawSection(n); this.ticking(); });
    return sec;
  }
  isOpen(n) { return this.dialogShown && !!this.sections.el?.[n]?.open; }
  // the page's drawer shown or not (dialogIn): hidden, nothing is drawn and the floor does not tick
  showDialog(shown) {
    if (this.dialogShown === !!shown) return;
    this.dialogShown = !!shown;
    for (const n of this.sections.names ?? []) this.drawSection(n);
    this.ticking();
  }

  sectionBody(n) {
    const d = this.dom;
    if (n === 'floor') {
      d.fsm = el('div', { class: 'fw-fsm' });
      this.fsm = new FsmView(d.fsm);
      d.dwell = el('p', { class: 'fw-dwell' }); d.trail = el('p', { class: 'fw-trail', title: 'the last floor states and how long each held' });
      d.costs = el('p', { class: 'fw-costs', title: "§3.2's expected costs of the system's actions now (observational)" });
      return el('div', { class: 'fw-body' }, d.fsm, d.dwell, d.trail, d.costs);
    }
    if (n === 'state') return el('div', { class: 'fw-body' }, d.stats = el('dl', { class: 'fw-stats' }));
    if (n === 'turns') {
      d.turns = el('tbody');
      return el('div', { class: 'fw-body fw-scroll' }, el('table', { class: 'fw-turns' },
        el('thead', {}, el('tr', {}, ['#', 'mode', 'heard', '1st tok', 'reply', 'dwell', 'end'].map(h => el('th', {}, h)))), d.turns));
    }
    d.events = el('div', { class: 'fw-events' });
    const save = el('button', { type: 'button', class: 'fw-link' }, 'save (JSONL)'), clear = el('button', { type: 'button', class: 'fw-link' }, 'clear');
    save.onclick = () => this.saveEvents();
    clear.onclick = () => { this.evFrom = this.core.bench.events.length; d.events.replaceChildren(); };
    return el('div', { class: 'fw-body' }, el('div', { class: 'fw-tools' }, save, clear), d.events);
  }

  // every dialog event, as the bench keeps it: the open sections follow
  drawEvent(e) {
    if (!this.sections?.names) return;
    if (e.kind === 'fsttm') { this.trail.push({ state: e.dst, t: e.t }); if (this.trail.length > 7) this.trail.shift(); if (this.isOpen('floor')) this.drawSection('floor'); }
    if (this.isOpen('events')) this.appendEvent(e);
    if (this.isOpen('turns') && ['llm', 'llm_aborted', 'speak', 'tts_end', 'cut'].includes(e.kind)) this.drawSection('turns');
    if (this.isOpen('state')) this.statsSoon();
  }
  appendEvent(e, follow = true) {
    const box = this.dom.events, end = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    const { t, kind, ...rest } = e;
    box.append(el('div', {}, `${(t / 1000).toFixed(2).padStart(7)} ${kind} ${JSON.stringify(rest)}`));
    while (box.children.length > 600) box.firstChild.remove();
    if (follow && end) box.scrollTop = box.scrollHeight;
  }
  statsSoon() { if (this.statsDue) return; this.statsDue = true; setTimeout(() => { this.statsDue = false; this.drawSection('state'); }, 250); }
  // the floor's dwell and costs move with time: a tick while the floor section is open
  ticking() {
    const want = this.isOpen('floor');
    if (want && !this.tick) this.tick = setInterval(() => this.drawSection('floor'), 200);
    if (!want && this.tick) { clearInterval(this.tick); this.tick = null; }
  }

  drawSection(n) {
    if (!this.isOpen(n)) return;
    const c = this.core, d = this.dom, f = c.dialog.fsttm, ms = v => v == null ? '—' : `${Math.round(v)}`;
    if (n === 'floor') {
      this.fsm.show(f.state, f.prev);
      d.dwell.textContent = `${f.state} for ${((f.now() - f.since) / 1000).toFixed(1)} s${f.prev ? ` · from ${f.prev}` : ''}`;
      d.trail.replaceChildren(...this.trail.flatMap((x, i) => {
        const until = i + 1 < this.trail.length ? this.trail[i + 1].t : null;
        const s = el('span', { class: i === this.trail.length - 1 ? 'fw-cur' : i === this.trail.length - 2 ? 'fw-prev' : '' }, until != null ? `${x.state} ${Math.round(until - x.t)} ms` : x.state);
        return i ? [' → ', s] : [s];
      }));
      const cost = f.systemActionsCost?.() ?? {};
      d.costs.textContent = 'system cost now: ' + Object.entries(cost).map(([k, v]) => `${k} ${v < 1e4 ? v.toFixed(1) : v.toExponential(1)}`).join(' · ');
    } else if (n === 'state') {
      const s = c.state(), m = c.bench.metrics(), lastCut = c.bench.events.findLast(e => e.kind === 'cut');
      const rows = [
        ['floor', s.floor], ['attention', s.wakeable ? s.attention : `${s.attention} (no wake word)`],
        ['mode', s.mode || '—'], ['reply', c.modes ? (c.modes.find(x => x.value === s.reply)?.label ?? s.reply) : s.reply], ['barge-in', s.bargeIn],
        ['recognizer', s.recognizer], ['voice', s.voice ? 'on' : 'off'], ['tts heard', c.tts.boundaries ? 'word boundaries' : 'estimated by time'],
        ...(c.backendName ? [['backend', c.backendName]] : []),
        ['', ''],
        ['turns', c.timings.length],
        ['barge-ins', `${m.cuts}${m.cuts ? ' · ' + Object.entries(m.byPolicy).map(([k, v]) => `${k} ${v}`).join(', ') : ''}`],
        ['onset→cut', m.onsetToCutP50 != null ? `p50 ${ms(m.onsetToCutP50)} · max ${ms(m.onsetToCutMax)} ms` : '—'],
        ['last cut', lastCut ? `${lastCut.why} · ${ms(lastCut.onsetToCutMs)} ms · "${(lastCut.heard ?? '').split(' ').slice(-4).join(' ')}"` : '—'],
        ['false cuts', m.falseCuts], ['echo drops', m.echoDrops],
        ['', ''],
        ['first token', m.llmFirstTokenP50 != null ? `p50 ${ms(m.llmFirstTokenP50)} ms` : '—'],
        ['reply time', m.llmFirstSentenceP50 != null ? `p50 ${ms(m.llmFirstSentenceP50)} ms` : '—'],
        ['FREEu dwell', m.freeuDwellP50 != null ? `p50 ${ms(m.freeuDwellP50)} ms` : '—'],
        ['tts', `${m.ttsSeconds} s spoken`],
      ];
      d.stats.replaceChildren(...rows.flatMap(([k, v]) => [el('dt', {}, k), el('dd', { title: String(v) }, String(v))]));
    } else if (n === 'turns') {
      d.turns.replaceChildren(...c.timings.slice(-200).map(t => el('tr', { class: t.end?.startsWith('cut') ? 'fw-cut' : '' },
        [t.n, t.mode, t.text, ms(t.first), ms(t.ms), ms(t.dwell), t.end].map((v, i) => el('td', i === 2 ? { title: t.text } : {}, v ?? '')))));
    } else if (n === 'events') {
      d.events.replaceChildren();
      for (const e of c.bench.events.slice(Math.max(this.evFrom, c.bench.events.length - 600))) this.appendEvent(e, false);
      d.events.scrollTop = d.events.scrollHeight;
    }
  }

  // the events file: every dialog event, timed (JSONL)
  saveEvents() {
    const a = el('a', { href: URL.createObjectURL(new Blob([this.core.bench.jsonl()], { type: 'application/jsonl' })), download: `fsttm-bench-${Date.now()}.jsonl` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  drawState(s) {
    const d = this.dom;
    d.floor.textContent = s.floor;
    d.floor.title = 'the floor';
    d.mode.textContent = s.mode;
    d.mode.hidden = !s.mode;
    d.mic.textContent = s.listening ? 'mic on' : 'mic';
    d.mic.setAttribute('aria-pressed', String(s.listening));
    d.voice.checked = s.voice;
    if (d.barge) {
      d.barge.value = s.bargeIn;
      d.attn.hidden = !s.wakeable;
      d.attn.textContent = s.attention === 'AWAKE' ? 'awake' : 'asleep';
      d.attn.setAttribute('aria-pressed', String(s.attention === 'AWAKE'));
      d.reply.hidden = !s.modes;
      if (s.modes) d.reply.value = s.reply;
    }
    if (this.isOpen?.('state')) this.statsSoon();
  }

  drawRow(r, how) {
    let li = this.rowEl.get(r.n);
    if (!li) {
      const who = r.who === 'person' ? (r.via === 'voice' ? 'heard' : r.via === 'text' ? 'typed' : 'said')
        : r.who === 'intent' ? 'intent' : r.who === 'agent' ? (r.narration ? 'unasked' : 'reply') : 'note';
      li = el('li', { class: `fw-row fw-${r.who}` },
        el('span', { class: 'fw-who' }, who), el(r.who === 'intent' ? 'code' : 'span', { class: 'fw-text' }, r.text),
        this.meta && r.meta ? el('small', { class: 'fw-meta' }, r.meta) : null,
        el('span', { class: 'fw-tags' }, r.tags.map(t => el('span', { class: 'fw-tag' }, t))));
      this.rowEl.set(r.n, li);
      const box = this.dom.log, end = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
      box.append(li);
      while (box.children.length > 300) { this.rowEl.delete(Number(box.firstChild.dataset.fwRow)); box.firstChild.remove(); }
      li.dataset.fwRow = r.n;                   // data-fw-row: a page's own data-* never meets the panel's
      if (end) box.scrollTop = box.scrollHeight;
    }
    if (how === 'tag') li.querySelector('.fw-tags').replaceChildren(...r.tags.map(t => el('span', { class: 'fw-tag' }, t)));
  }
}
