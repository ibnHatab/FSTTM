// bench.js — every dialog event with performance.now(), exportable as
// JSONL (the panel's events section saves it), and the barge-in metrics computed from them: onset→cut
// latency, false cuts, echo drops, user words lost at the cut, FREEu
// dwell, LLM first-token and first-sentence times.

const words = s => (s ?? '').toLowerCase().match(/[a-z0-9']+/g) ?? [];
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null; };

export class Bench {
  // max: the events kept (the oldest go first; a long session stays bounded — the numbers
  // are kept as the events come, so they count every event, kept or not)
  constructor({ now = () => performance.now(), sink = null, max = 20000 } = {}) {
    this.now = now;
    this.t0 = now();
    this.events = [];
    this.sink = sink; // (event) => void, e.g. the page's log pane
    this.max = max;
    this.m = { cuts: 0, byPolicy: {}, onsetToCut: [], confirmed: 0, pendingFalse: [], pendingLost: [], lost: 0,
      openSpeaks: [], ttsMs: 0, echoDrops: 0, dwell: [], llmFirst: [], llmMs: [] };
  }

  log = (kind, fields = {}) => {
    const e = { t: Math.round((this.now() - this.t0) * 10) / 10, kind, ...fields };
    this.events.push(e);
    if (this.events.length > this.max) this.events.splice(0, this.events.length - this.max);
    this.count(e);
    this.sink?.(e);
    return e;
  };

  jsonl() { return this.events.map(e => JSON.stringify(e)).join('\n') + '\n'; }

  // count: the numbers, one event at a time (metrics() reads them: no pass over the events)
  count(e) {
    const m = this.m;
    if (e.kind === 'final') {
      // a cut is false when no final with words follows it within 4 s; the words its interim had and
      // the final after it has not are lost at the cut
      if (words(e.text).length) {
        const keep = m.pendingFalse.filter(c => !(e.t > c.t && e.t - c.t < 4000));
        m.confirmed += m.pendingFalse.length - keep.length; m.pendingFalse = keep;
      }
      const after = m.pendingLost.filter(c => e.t > c.t), have = new Set(words(e.text));
      for (const c of after) m.lost += words(c.interim).filter(w => !have.has(w)).length;
      m.pendingLost = m.pendingLost.filter(c => !(e.t > c.t));
    }
    if (e.kind === 'tts_end' || e.kind === 'cut') {   // a speak ends at the first tts_end or cut after it
      for (const s of m.openSpeaks) if (e.t > s) m.ttsMs += e.t - s;
      m.openSpeaks = m.openSpeaks.filter(s => !(e.t > s));
    }
    if (e.kind === 'cut') {
      m.cuts++; m.byPolicy[e.why] = (m.byPolicy[e.why] ?? 0) + 1;
      if (e.onsetToCutMs != null) m.onsetToCut.push(e.onsetToCutMs);
      m.pendingFalse.push(e);
      if (e.interim) m.pendingLost.push(e);
    } else if (e.kind === 'speak') {
      m.openSpeaks.push(e.t);
      if (e.freeuDwell != null) m.dwell.push(e.freeuDwell);
    } else if (e.kind === 'echo') m.echoDrops++;
    else if (e.kind === 'llm') { if (e.firstMs != null) m.llmFirst.push(e.firstMs); m.llmMs.push(e.ms); }
  }

  metrics() {
    const m = this.m, falseCuts = m.cuts - m.confirmed;
    return {
      cuts: m.cuts, byPolicy: { ...m.byPolicy },
      onsetToCutP50: pct(m.onsetToCut, 0.5), onsetToCutMax: m.onsetToCut.length ? Math.max(...m.onsetToCut) : null,
      falseCuts, falseCutsPerTtsMin: m.ttsMs ? falseCuts / (m.ttsMs / 60000) : null,
      echoDrops: m.echoDrops,
      userWordsLostAtCut: m.lost,
      freeuDwellP50: pct(m.dwell, 0.5),
      llmFirstTokenP50: pct(m.llmFirst, 0.5),
      llmFirstSentenceP50: pct(m.llmMs, 0.5),
      ttsSeconds: Math.round(m.ttsMs / 100) / 10,
    };
  }
}

// metricsOf: the same numbers from a list of events in one go (a saved JSONL read back; the tests'
// reference for Bench's counting)
export function metricsOf(ev) {
  const cuts = ev.filter(e => e.kind === 'cut');
  // a cut is false when no final with words follows it within 4 s
  const falseCuts = cuts.filter(c => !ev.some(e => e.kind === 'final' && e.t > c.t && e.t - c.t < 4000 && words(e.text).length));
  const lost = cuts.map(c => {
    const f = ev.find(e => e.kind === 'final' && e.t > c.t);
    if (!f || !c.interim) return null;
    const have = new Set(words(f.text));
    return words(c.interim).filter(w => !have.has(w)).length;
  }).filter(x => x !== null);
  const ttsMs = ev.filter(e => e.kind === 'speak').reduce((acc, s) => {
    const end = ev.find(e => (e.kind === 'tts_end' || e.kind === 'cut') && e.t > s.t);
    return acc + ((end?.t ?? s.t) - s.t);
  }, 0);
  const llm = ev.filter(e => e.kind === 'llm');
  const onsetToCut = cuts.map(c => c.onsetToCutMs).filter(x => x != null);
  const dwell = ev.filter(e => e.kind === 'speak' && e.freeuDwell != null).map(e => e.freeuDwell);
  return {
    cuts: cuts.length, byPolicy: cuts.reduce((m, c) => (m[c.why] = (m[c.why] ?? 0) + 1, m), {}),
    onsetToCutP50: pct(onsetToCut, 0.5), onsetToCutMax: onsetToCut.length ? Math.max(...onsetToCut) : null,
    falseCuts: falseCuts.length, falseCutsPerTtsMin: ttsMs ? falseCuts.length / (ttsMs / 60000) : null,
    echoDrops: ev.filter(e => e.kind === 'echo').length,
    userWordsLostAtCut: lost.reduce((a, b) => a + b, 0),
    freeuDwellP50: pct(dwell, 0.5),
    llmFirstTokenP50: pct(llm.map(e => e.firstMs).filter(x => x != null), 0.5),
    llmFirstSentenceP50: pct(llm.map(e => e.ms), 0.5),
    ttsSeconds: Math.round(ttsMs / 100) / 10,
  };
}
