// fsttm.js — the N09-1071 finite-state turn-taking model, interpreted from
// fsttm.json (the machine as data: one table, any interpreter).
// States are floor OWNERSHIP, not speech. lib/fixtures/fsttm-cases.json is
// its test (test/run.mjs).

export class FSTTM {
  // table: the parsed fsttm.json; now: a clock in ms (performance.now)
  constructor(table, { now = () => performance.now(), onChange = null } = {}) {
    this.t = table;
    this.now = now;
    this.onChange = onChange;
    this.state = table.boot;
    this.prev = null;
    this.since = now();
    this.system = 'W'; // current system action, remapped: W|K
    this.user = 'W';   // current user action, remapped: W|K
  }

  isSystem() { return this.t.system_holds.includes(this.state); }
  isUser() { return this.t.user_holds.includes(this.state); }

  // trigger fires the transition keyed "<system>_<user>" from the current
  // state; throws on a pair the paper does not allow.
  trigger(event) {
    const tr = this.t.transitions.find(x => x.event === event && x.src === this.state);
    if (!tr) throw new Error(`fsttm: invalid event ${event} for state ${this.state}`);
    this.prev = this.state;
    this.state = tr.dst;
    this.since = this.now();
    if (this.onChange) this.onChange(event, tr.src, tr.dst);
    return tr.dst;
  }

  systemAction(a) {
    const s = this.trigger(`${a}_${this.user}`);
    this.system = this.t.remap[a];
    return s;
  }

  userAction(a) {
    const s = this.trigger(`${this.system}_${a}`);
    this.user = this.t.remap[a];
    return s;
  }

  // tryAction is systemAction/userAction that answers false instead of
  // throwing (the dialog ignores self-loops and stale actions, as Go does).
  tryAction(actor, a) {
    try { (actor === 'system' ? this.systemAction(a) : this.userAction(a)); return true; } catch { return false; }
  }

  // systemActionsCost is §3.2's action → expected cost for the current state
  // and dwell, from the JSON constants (observational, as in the harness).
  systemActionsCost() {
    const c = this.t.cost, tau = this.now() - this.since, cO = Math.exp((tau + 100) / 1000);
    if (this.isSystem()) return { K: c.p_b * cO, R: (1 - c.p_b) * c.c_s };
    let cG, pF;
    if (this.state === 'FREEu') [cG, pF] = [c.c_g_pause * tau, c.p_f_pause];
    else if (this.state === 'USER') [cG, pF] = [c.c_g_speech, c.p_f_speech];
    else [cG, pF] = [c.c_g_free_s, c.p_f_free_s];
    return { W: pF * cG, G: (1 - pF) * c.c_u };
  }
}
