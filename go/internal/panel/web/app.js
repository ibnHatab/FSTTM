// The engine's page: the dialog (fsttm-web's Panel) on the left, the ENVIRONMENT and REASONING streams and the
// page's floor machine in the shell's docks. The engine owns the mic, the recognition, the LLM and the voice: the
// panel here shows the conversation (what the NUC heard, her replies) and takes typed turns (nuc/say), voice off.
import { Panel } from './vendor/panel.js';

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const hm = ts => ts ? new Date(ts * 1000).toTimeString().slice(0, 8) : '--:--:--';

// ── the socket ──
let ws = null;
const pending = new Map();          // a typed turn's id → resolve(reply)
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('#conn').textContent = 'live'; };
  ws.onclose = () => { $('#conn').textContent = 'reconnecting…'; setTimeout(connect, 2000); };
  ws.onmessage = m => { try { frame(JSON.parse(m.data)); } catch (e) { console.error(e); } };
}

// the panel's backend: a typed turn goes to the engine, the reply comes back as nuc/reply with its id
const backend = {
  onFrame: null,
  async *turn(id, text) {
    const reply = new Promise(res => pending.set(id, res));
    ws?.send(JSON.stringify({ ev: 'nuc/say', id, text }));
    const words = await Promise.race([reply, new Promise(res => setTimeout(() => res(''), 60000))]);
    pending.delete(id);
    if (words) yield words;
  },
  cut() {}, heard() {},
};

const panel = new Panel($('#dialog'), {
  backend, table: await (await fetch('vendor/fsttm.json')).json(), title: '', voice: false,
  micButton: false, sections: ['floor', 'state', 'events'], dialogIn: $('#floorpart'), hostParam: null,
  storeKey: 'nina',
});
window.panel = panel;

// ── the docks ──
const docks = MetisShell.docks({ names: ['env', 'rsn', 'floor'], key: 'nina.docks.v1', fold: false,
  defaults: { env: { shown: true, open: true }, rsn: { shown: true, open: true }, floor: { shown: false, open: true } },
  onChange: st => panel.showDialog(!!st.floor?.shown) });
docks.apply();
MetisShell.splitter();
panel.showDialog(!!docks.state.floor?.shown);

// ── the streams ──
const env = new Map();
function drawEnv(fresh) {
  const b = $('#env');
  if (!env.size) { b.innerHTML = '<p class="empty">no state yet</p>'; return; }
  b.innerHTML = [...env.entries()].sort(([a], [c]) => a.localeCompare(c)).map(([k, e]) =>
    `<div class="row${k === fresh ? ' fresh' : ''}" title="${esc(k)}"><span class="t">${hm(e.ts)}</span><span class="v2">${esc(e.text)}</span></div>`).join('');
}
function reason(f, live) {
  const b = $('#rsn');
  b.querySelector('.empty')?.remove();
  const d = document.createElement('div');
  const tool = f.ev === 'nuc/tool';
  const text = tool ? `${f.name}(${f.args}) → ${f.answer}` : f.text;
  d.className = `row${live ? ' fresh' : ''}${tool ? ' tool' : ''}${/refused|cannot|can't|error/i.test(text) ? ' refused' : ''}`;
  d.innerHTML = `<span class="t">${hm(f.ts)}</span><span class="v2">${esc(text)}</span>`;
  b.prepend(d);
  while (b.children.length > 200) b.lastChild.remove();
}
// the engine's floor: its machine is the one that holds the turn (the NUC's mic and voice); the page's
// own floor machine (fsttm-web's, drawn in the FLOOR dock) is moved to the engine's state on every
// transition, so the diagram, the dwell, the trail and the costs are the engine's
function engineFloor(dst, src, event) {
  const f = panel.core.dialog?.fsttm;
  if (!f || !dst || f.state === dst) return;
  f.prev = src ?? f.state;
  f.state = dst;
  f.since = f.now();
  panel.core.event('fsttm', { event, src: f.prev, dst, by: 'engine' });
}
// every engine event in the FLOOR dock's events list (fsttm-web's bench keeps and times them)
function engineEvent(f) {
  const { ev, kind, t, ...rest } = f;
  if (kind === 'fsttm') { engineFloor(rest.dst, rest.src, rest.event); return; }
  panel.core.event(`nuc.${kind}`, rest);
}
function state(s) {
  if (s.floor) { $('#floor').textContent = s.floor; engineFloor(s.floor); }
  if (s.attention) $('#attn').textContent = s.attention;
  if (s.dialog) $('#dlg').textContent = s.dialog;
}
// the conversation the NUC heard: rows in the panel (a typed turn's reply resolves its own turn)
function said(f) {
  if (f.ev === 'nuc/heard') panel.core.row({ who: 'person', text: f.text, via: 'voice', meta: 'heard on the NUC' });
  else if (f.ev === 'nuc/asleep') panel.core.row({ who: 'note', text: `asleep, ignored: “${f.text}” (say “hey Nina” first)` });
  else if (f.ev === 'nuc/reply') {
    if (f.id && pending.has(f.id)) { pending.get(f.id)(f.text); return; }
    panel.core.row({ who: 'agent', text: f.text, meta: 'her reply, spoken on the NUC' });
  }
}
function frame(f) {
  switch (f.ev) {
    case 'nuc/snapshot':
      env.clear(); for (const e of f.env || []) env.set(e.key, e); drawEnv();
      $('#rsn').innerHTML = (f.reasoning || []).length ? '' : '<p class="empty">nothing concluded yet</p>';
      for (const r of f.reasoning || []) reason(r, false);
      for (const c of f.conversation || []) if (!c.id) said(c);
      state(f.state || {});
      return;
    case 'nuc/env': if (f.text == null) env.delete(f.key); else env.set(f.key, f); drawEnv(f.key); return;
    case 'nuc/reasoning': case 'nuc/tool': reason(f, true); return;
    case 'nuc/heard': case 'nuc/asleep': case 'nuc/reply': said(f); return;
    case 'nuc/state': state(f); return;
    case 'nuc/event': engineEvent(f); return;
  }
}
connect();
document.body.dataset.ready = '1';
