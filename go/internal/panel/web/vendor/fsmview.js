// fsmview.js — the floor machine drawn as the Python TUI draws it (the panel's floor section; FSTTM
// fsttm/tui.py _fsm_diagram): FREEs on top, SYSTEM · BOTHs · USER across,
// BOTHu and FREEu below, joined by the same branch bars; the current state
// green, the one just left blue, the rest dim.
//
//          FREEs (gap after system)
//            │
//     ┌──────┼──────┐
//     │      │      │
//  SYSTEM  BOTHs  USER
//     │      │      │
//     └──────┼──────┘
//            │
//          BOTHu
//            │
//          FREEu (gap after user)

const L = 70, C = 160, R = 250;  // the three columns
const ROW = { FREEs: 18, top: 62, mid: 108, bot: 154, BOTHu: 196, FREEu: 240 };
const NODES = { FREEs: [C, ROW.FREEs], SYSTEM: [L, ROW.mid], BOTHs: [C, ROW.mid], USER: [R, ROW.mid], BOTHu: [C, ROW.BOTHu], FREEu: [C, ROW.FREEu] };
const NOTE = { FREEs: '(gap after system)', FREEu: '(gap after user)' };
const GAP = 12; // the lines stop this far from a label
const SVG = 'http://www.w3.org/2000/svg';

const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.append(e);
  return e;
};

export class FsmView {
  constructor(root) {
    this.svg = el('svg', { viewBox: '0 0 360 258', class: 'fsm', role: 'img', 'aria-label': 'floor state' }, root);
    const lines = [
      [C, ROW.FREEs + GAP, C, ROW.top],            // FREEs stem
      [L, ROW.top, R, ROW.top],                    // ┌──┼──┐
      [L, ROW.top, L, ROW.mid - GAP], [C, ROW.top, C, ROW.mid - GAP], [R, ROW.top, R, ROW.mid - GAP],
      [L, ROW.mid + GAP, L, ROW.bot], [C, ROW.mid + GAP, C, ROW.bot], [R, ROW.mid + GAP, R, ROW.bot],
      [L, ROW.bot, R, ROW.bot],                    // └──┼──┘
      [C, ROW.bot, C, ROW.BOTHu - GAP],            // stem to BOTHu
      [C, ROW.BOTHu + GAP, C, ROW.FREEu - GAP],    // BOTHu to FREEu
    ];
    for (const [x1, y1, x2, y2] of lines) el('line', { x1, y1, x2, y2, class: 'fsm-line' }, this.svg);
    this.nodes = {};
    for (const [name, [x, y]] of Object.entries(NODES)) {
      const t = el('text', { x, y, class: 'fsm-node', 'text-anchor': 'middle', 'dominant-baseline': 'middle' }, this.svg);
      t.textContent = name;
      if (NOTE[name]) el('text', { x: x + 36, y, class: 'fsm-note', 'dominant-baseline': 'middle' }, this.svg).textContent = NOTE[name];
      this.nodes[name] = t;
    }
  }

  // show: current green, previous blue (the transition just taken), rest dim
  show(state, prev = null) {
    for (const [n, t] of Object.entries(this.nodes)) {
      t.classList.toggle('cur', n === state);
      t.classList.toggle('prev', n === prev && n !== state);
    }
  }
}
