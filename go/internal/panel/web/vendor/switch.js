// switch.js — one on/off switch with its indicator, for a page's header row (switch.css): a
// button with role switch, a light, its label and the track; the light and the track say the
// state (no word: words {on, off, …} adds one). The light: off (dim), busy (amber, blinking:
// opening), on (the accent), error (red); ping() flashes it a moment (the mic: a word heard).
//
//   const sw = toggleSwitch({ label: 'mic', onChange: want => … });  header.append(sw.el);
//   sw.set('on' | 'off' | 'busy' | 'error', title?);  sw.ping();  sw.disable(why);
//
// micSwitch(panel) is the dialog panel's mic as one: it follows the panel (subscribe) and
// toggles it; the light flashes on each word the recognizer hears. voiceSwitch(panel) is the
// panel's voice (its replies spoken: PanelCore.setVoice) as one, following the panel; Shift+8
// turns it on and off outside a text field (speakKey; null: no key), kept per page (store).
// narratorLabel(narrator) is the narrator (lib/narrator.js), always on: its name, coloured while
// it reads (green), paused (amber) or held for the dialog (cyan); a click is its key's press; the
// sentence and n of N in its title. voiceWidget(panel, {narrator}) is the header's one card for
// the voice: [● mic ⟦switch⟧ ● voice ⟦switch⟧ narrator] (the user, 2026-11-07: the voice switch in
// the widget, not a page's own); F9 turns the mic on and off (micKey; null: no key), Shift+8 the
// voice (speakKey), F8 is the narrator's (key); voice: false leaves the voice switch out.

const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.append(...kids);
  return e;
};

export function toggleSwitch({ label, title = '', onChange = () => {}, words = null } = {}) {
  const light = h('span', { class: 'fw-sw-light', 'aria-hidden': 'true' });
  const word = words ? h('span', { class: 'fw-sw-word' }) : null;
  const el = h('button', { class: 'fw-switch', type: 'button', role: 'switch', 'aria-checked': 'false', title, 'aria-label': label },
    light, h('span', { class: 'fw-sw-label' }, label), h('span', { class: 'fw-sw-track', 'aria-hidden': 'true' }, h('span', { class: 'fw-sw-knob' })), ...(word ? [word] : []));
  let state = 'off', timer = null;
  const sw = {
    el,
    get state() { return state; },
    get on() { return state === 'on'; },
    set(s, why) {
      state = s;
      el.dataset.state = s;
      el.setAttribute('aria-checked', String(s === 'on'));
      if (word) word.textContent = words[s] ?? s;
      if (why != null) el.title = why;
    },
    ping(ms = 220) {
      el.classList.add('fw-sw-ping');
      clearTimeout(timer);
      timer = setTimeout(() => el.classList.remove('fw-sw-ping'), ms);
    },
    disable(why) { el.disabled = true; sw.set('off', why); },
  };
  el.onclick = () => onChange(state !== 'on');
  sw.set('off');
  return sw;
}

// the narrator, always on: its name, coloured by its state (reading, paused, yielding; stopped plain, dim until
// text is selected); the light-less label flashes as it moves to the next sentence; a click is the key's press
export function narratorLabel(narrator, { label = 'narrator', key = 'F8' } = {}) {
  const help = `the narrator: select text, then ${key} (or a click) reads it · again pauses · again resumes; hold or double ${key}: stop`;
  const el = h('button', { class: 'fw-narr', type: 'button', title: help }, label);
  el.onclick = () => narrator.press();
  const prev = narrator.onchange;
  let at = null, timer = null;
  narrator.onchange = s => {
    prev?.(s);
    el.dataset.state = s.state;
    el.dataset.ready = s.n ? '1' : '0';
    el.title = help + (s.n ? ` · ${s.state} · ${Math.min(s.i + 1, s.n)} of ${s.n}: ${s.sentence}` : ' · nothing selected');
    el.setAttribute('aria-label', `narrator, ${s.state}`);
    if (s.state === 'reading' && at !== null && s.i !== at) {
      el.classList.add('fw-sw-ping'); clearTimeout(timer); timer = setTimeout(() => el.classList.remove('fw-sw-ping'), 220);
    }
    at = s.i;
  };
  narrator.onchange(narrator.snapshot());
  return { el };
}

// the voice in a header, one card: [● mic ⟦switch⟧ ● voice ⟦switch⟧ narrator] (the narrator with a narrator only)
export function voiceWidget(panel, { narrator = null, key = 'F8', micKey = 'F9', voice = true, speakKey = 'Shift+8', store = null, target = globalThis.document, ...mic } = {}) {
  const m = micSwitch(panel, { key: micKey, target, ...mic }), n = narrator ? narratorLabel(narrator, { key }) : null;
  const v = voice ? voiceSwitch(panel, { key: speakKey, store, target }) : null;
  const el = h('span', { class: 'fw-voicebar', role: 'group', 'aria-label': 'the voice' }, m.el, ...(v ? [v.el] : []), ...(n ? [n.el] : []));
  return { el, mic: m, voice: v, narrator: n };
}

// a key as a page names it: 'F9', 'Shift+8' (the digit's key, whatever the layout prints on it)
export function keyMatch(name, e) {
  const parts = String(name).split('+'), k = parts.pop(), mods = new Set(parts.map(p => p.toLowerCase()));
  if (!!e.shiftKey !== mods.has('shift') || e.ctrlKey || e.altKey || e.metaKey) return false;
  return /^\d$/.test(k) ? e.code === `Digit${k}` : e.key === k;
}
const typing = t => !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName ?? ''));

// the panel's voice: its replies spoken or not; kept per page under store (localStorage), a key outside a text field
export function voiceSwitch(panel, { label = 'voice', title = 'the voice: replies spoken while on', key = 'Shift+8', store = null, target = globalThis.document } = {}) {
  if (key) title += ` (${key})`;
  const ls = globalThis.localStorage;
  if (store && ls) { try { const v = ls.getItem(store); if (v != null) panel.core.setVoice(v === '1'); } catch { /* the default */ } }
  const set = on => { panel.core.setVoice(on); if (store && ls) try { ls.setItem(store, on ? '1' : '0'); } catch { /* this page only */ } };
  const sw = toggleSwitch({ label, title, onChange: set });
  sw.set(panel.core.voice ? 'on' : 'off');
  if (key && target) target.addEventListener('keydown', e => {
    if (e.repeat || typing(e.target) || !keyMatch(key, e)) return;
    e.preventDefault();
    set(!panel.core.voice);
  });
  panel.subscribe(({ state }) => { if (state) sw.set(state.voice ? 'on' : 'off'); });
  return Object.assign(sw, { toggle: () => set(!panel.core.voice), setVoice: set });
}

// the panel's mic: on / off / opening, the light flashing on what the recognizer hears
const HEARD = new Set(['onset', 'interim', 'final']);
export function micSwitch(panel, { label = 'mic', title = 'the microphone: the dialog listens while on', key = null, target = globalThis.document } = {}) {
  if (key) title += ` (${key})`;
  const sw = toggleSwitch({ label, title, onChange: () => panel.toggleMic() });
  // the key: a press is a click on the switch (a user gesture: the browser lets it open the mic)
  if (key && target && panel.Recognizer) target.addEventListener('keydown', e => {
    if (e.key !== key || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    e.preventDefault();
    panel.toggleMic();
  });
  if (!panel.Recognizer) { sw.disable('no SpeechRecognition in this browser'); return sw; }
  panel.subscribe(({ state, event }) => {
    if (state) sw.set(panel.opening ? 'busy' : state.listening ? 'on' : 'off', `${title} · recognizer: ${state.recognizer}`);
    else if (event?.kind === 'rec_error') sw.set(panel.core.listening ? 'on' : 'error', `microphone: ${event.error ?? 'error'}`);
    else if (HEARD.has(event?.kind) && sw.on) sw.ping();
  });
  return sw;
}
