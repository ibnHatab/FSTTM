"""
Rich-based 3-panel TUI for the fsttm voice server.

Layout
------
  ┌───────────────────────────────┬───────────────────┐
  │ Chat            (top-left)     │ State / Perf      │
  │  [user] … / [assistant] …     │  (right, full     │
  ├───────────────────────────────┤   height,         │
  │ Intents         (bottom-left) │   refreshable)    │
  │  JSON commands → domain backend│                   │
  └───────────────────────────────┴───────────────────┘

Design
------
The voice server runs on an asyncio loop (cyclotron). This module keeps a single
mutable ``TUIState`` that the reactive streams feed via ``attach()`` (replacing
the old stdout prints), and drives a ``rich.live.Live`` that re-renders the whole
layout on a timer scheduled on the same loop — so there is no second thread and
no lock contention with the reactive graph.

Graceful degradation: if ``rich`` is not installed, ``FsttmTUI.available`` is
False and the server falls back to plain stdout logging.
"""
import os
import sys
import time
from collections import deque

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except Exception:                       # pragma: no cover - depends on host
    _RICH = False


# ── shared STT perf holder ────────────────────────────────────────────────────
# whisper.py updates this in its transcribe thread; the right panel reads it.
# A plain dict avoids importing Rich into the STT driver.
STT_PERF = {"ms": 0.0, "audio_s": 0.0, "rtf": 0.0, "n": 0}
# Intent two-pass timing — JSON (grammar-constrained) vs text (spoken ack) gen.
# Surfaced in the State·Perf panel so a latency regression is visible at a glance.
INTENT_PERF = {"json_ms": 0.0, "text_ms": 0.0, "n": 0}

# True while a Live TUI owns the screen — other modules check this to suppress
# stray prints that would corrupt the alt-screen render.
_ACTIVE = [False]


def active():
    return _ACTIVE[0]


def record_stt_perf(ms, audio_s, rtf):
    STT_PERF["ms"] = ms
    STT_PERF["audio_s"] = audio_s
    STT_PERF["rtf"] = rtf
    STT_PERF["n"] += 1


def record_intent_perf(json_ms, text_ms):
    """JSON-pass vs text-pass generation time for the last intent (regression
    watch — split so a slowdown in either is obvious)."""
    INTENT_PERF["json_ms"] = json_ms
    INTENT_PERF["text_ms"] = text_ms
    INTENT_PERF["n"] += 1


# ── state ─────────────────────────────────────────────────────────────────────

_MAX_CHAT = 500
_MAX_INTENT = 200
_MAX_NOTES = 12


class TUIState:
    """All data shown by the TUI. Mutated by attach()'d subscriptions."""

    def __init__(self):
        self.started = time.monotonic()
        self.chat = deque(maxlen=_MAX_CHAT)        # list[(role, text, ts)]
        self.intents = deque(maxlen=_MAX_INTENT)   # list[(json, voice, ts)]
        self.notes = deque(maxlen=_MAX_NOTES)      # list[(level, text, ts)] right-panel log

        # right-panel live fields
        self.fsm_state = "USER"
        self.fsm_prev = ""
        self.fsm_last_action = ""
        self.intent_mode = False
        self.soft_duck = True
        self.attention_enabled = False
        self.attention_state = "OFF"   # OFF | ASLEEP | AWAKE
        self.aec = "?"
        self.domain_status = None   # (label, ok) from the domain dispatcher

        # narrator
        self.ckpt_cur = -1
        self.ckpt_total = 0
        self.last_barge = ""

        self.turns = 0          # completed user→assistant exchanges
        self.barge_ins = 0

    # --- mutators (called from reactive subscriptions / server prints) --------
    def add_user(self, text):
        self.chat.append(("user", text, time.monotonic()))
        self.turns += 1

    def add_assistant(self, text):
        self.chat.append(("assistant", text, time.monotonic()))

    def add_intent(self, intent_json, voice):
        # kind 'domain' → JSON command + spoken text (magenta)
        self.intents.append(("domain", str(intent_json), voice, time.monotonic()))

    def add_system_intent(self, text, action):
        # kind 'system' → attention classifier decision for an utterance.
        self.intents.append(("system", action, text, time.monotonic()))

    def note(self, text, level="info"):
        self.notes.append((level, text, time.monotonic()))

    def set_fsm(self, state, prev="", action=""):
        self.fsm_state = state
        self.fsm_prev = prev
        self.fsm_last_action = action


# ── rendering ─────────────────────────────────────────────────────────────────

_ROLE_STYLE = {"user": "bold green", "assistant": "bold blue"}
_LEVEL_STYLE = {"info": "dim", "warn": "yellow", "error": "bold red",
                "good": "green"}
_SYSTEM_STATES = {"SYSTEM", "BOTHs", "BOTHu"}


def _hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _wrapped_rows(text, width):
    """How many terminal rows `text` occupies when wrapped to `width` columns.
    A chat message is one logical entry but wraps to several rows — this is what
    the height budget must count, not the entry count."""
    if width <= 0:
        return 1
    rows = 0
    for line in text.split("\n"):
        rows += max(1, -(-len(line) // width))   # ceil(len/width)
    return rows


def _tail_by_rows(entries, height, width):
    """Keep the most recent entries whose cumulative WRAPPED-row count fits
    `height`, so the newest message stays fully visible (Rich crops overflow
    from the bottom otherwise). entries: list of (tag, style, text)."""
    if not height or height <= 0:
        return entries
    out, used = [], 0
    for entry in reversed(entries):           # newest first
        rows = _wrapped_rows(entry[0] + entry[2], width)
        if used + rows > height and out:      # keep at least the newest entry
            break
        out.append(entry)
        used += rows
    out.reverse()
    return out


def _chat_panel(state, height=0, width=0):
    entries = []
    for role, text, ts in state.chat:
        tag = "[user] " if role == "user" else "[assistant] "
        entries.append((tag, _ROLE_STYLE.get(role, ""), text))
    body = Text()
    for tag, style, text in _tail_by_rows(entries, height, width):
        body.append(tag, style=style)
        body.append(text + "\n")
    if not state.chat:
        body = Text("…waiting for speech…", style="dim italic")
    return Panel(body, title="Chat", title_align="left", border_style="cyan")


def _intent_panel(state, height=0, width=0):
    # Build each intent's lines, then keep the most recent entries that fit the
    # height by WRAPPED rows (a long JSON wraps to several rows).
    _ACTION_STYLE = {"sleep": "yellow", "mute": "yellow", "command": "green"}
    entries = []   # each is a list of (text, style) line-tuples
    for kind, primary, secondary, ts in state.intents:
        ts_s = time.strftime("%H:%M:%S", time.localtime(ts)) + "  "
        if kind == "system":
            # primary = action, secondary = the utterance that was classified
            ent = [(ts_s + "sys: " + primary,
                    _ACTION_STYLE.get(primary, "magenta"))]
            if secondary:
                ent.append(("   ↳ " + repr(secondary), "dim"))
        else:  # domain
            ent = [(ts_s + primary, "magenta")]
            if secondary:
                ent.append(("   ↳ " + repr(secondary), "dim"))
        entries.append(ent)

    kept, used = [], 0
    budget = height if (height and height > 0) else 10 ** 9
    for ent in reversed(entries):
        rows = sum(_wrapped_rows(t, width) for t, _ in ent)
        if used + rows > budget and kept:
            break
        kept.append(ent)
        used += rows
    kept.reverse()

    body = Text()
    for ent in kept:
        for t, style in ent:
            body.append(t + "\n", style=style)
    if not state.intents:
        body = Text("…no intents yet…", style="dim italic")
    return Panel(body, title="Intents (domain · sys)", title_align="left",
                 border_style="magenta")


# Turn-taking FSM diagram. Two "gap" states (FREEs/FREEu) bracket the three
# active-floor states; BOTHu is the user-initiated overlap below. Matches the
# real transition topology in fsttm.Model.
#   FREEs  → SYSTEM / BOTHs / USER → BOTHu → FREEu  (and back up)
_FSM_NODES = ("FREEs", "SYSTEM", "BOTHs", "USER", "BOTHu", "FREEu")
# (label, row, col) placement on the diagram grid.
_FSM_LAYOUT = [
    ("FREEs",  0, 1),
    ("SYSTEM", 2, 0), ("BOTHs", 2, 1), ("USER", 2, 2),
    ("BOTHu",  4, 1),
    ("FREEu",  5, 1),
]


def _node_style(node, state):
    if node == state.fsm_state:
        return "bold green"          # current
    if node == state.fsm_prev:
        return "bold blue"           # just left (shows the transition)
    return "dim"


def _fsm_diagram(state):
    """ASCII turn-taking diagram, current state green, previous blue.

    Three fixed-width node columns keep the branch bars (│ ┼ │) aligned exactly
    over SYSTEM / BOTHs / USER regardless of label width:

        col centres:  L=4   C=11   R=18      (0-indexed)
    """
    s = state
    d = "dim"

    # Three 7-wide cells (label centred) concatenated after a 1-space indent put
    # the column centres at fixed cols 4, 11, 18 — so the branch bars below align
    # exactly over SYSTEM / BOTHs / USER regardless of label width.
    def cell(label, w=7):
        return f"{label:^{w}}"

    INDENT = " "                              # shifts the row so centres = 4/11/18
    node_row = Text.assemble(
        (INDENT, d),
        (cell("SYSTEM"), _node_style("SYSTEM", s)),
        (cell("BOTHs"),  _node_style("BOTHs", s)),
        (cell("USER"),   _node_style("USER", s)),
    )
    #             0    4     11    18
    branch_top = "    ┌──────┼──────┐"        # ┌ at 4, ┼ at 11, ┐ at 18
    bars       = "    │      │      │"        # bars at 4, 11, 18
    branch_bot = "    └──────┼──────┘"
    stem       = "           │"              # single bar at centre col 11

    # FREEs / BOTHu / FREEu sit on the centre column → centre their 7-cell on
    # col 11, i.e. cell starts at col 8 (8 leading spaces before the cell).
    centre_pad = " " * 8

    lines = [
        Text.assemble((centre_pad, d), (cell("FREEs"), _node_style("FREEs", s)),
                      (" (gap after system)", d)),
        Text(stem, style=d),
        Text(branch_top, style=d),
        Text(bars, style=d),
        node_row,
        Text(bars, style=d),
        Text(branch_bot, style=d),
        Text(stem, style=d),
        Text.assemble((centre_pad, d), (cell("BOTHu"), _node_style("BOTHu", s))),
        Text(stem, style=d),
        Text.assemble((centre_pad, d), (cell("FREEu"), _node_style("FREEu", s)),
                      (" (gap after user)", d)),
    ]
    return Group(*lines)


def _state_panel(state):
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="dim")
    t.add_column()

    t.add_row("uptime", _hms(time.monotonic() - state.started))
    t.add_row("mode", "intent" if state.intent_mode else "chat")
    if state.attention_enabled:
        astyle = {"AWAKE": "bold green", "ASLEEP": "yellow"}.get(
            state.attention_state, "dim")
        t.add_row("attention", Text(state.attention_state, style=astyle))
    t.add_row("AEC", str(state.aec))
    t.add_row("soft_duck", "on" if state.soft_duck else "off")

    label, ok = state.domain_status or (None, False)
    dom = Text(label or "disabled", style="green" if ok else "dim")
    t.add_row("domain", dom)

    t.add_row("", "")
    t.add_row("turns", str(state.turns))
    t.add_row("barge-ins", str(state.barge_ins))

    if state.ckpt_total:
        t.add_row("narrator", f"ckpt {state.ckpt_cur}/{state.ckpt_total - 1}")
    if state.last_barge:
        t.add_row("last barge", Text(state.last_barge, style="dim"))

    p = STT_PERF
    t.add_row("", "")
    t.add_row("STT", f"{p['ms']:.0f}ms / {p['audio_s']:.1f}s")
    rtf_style = "green" if p["rtf"] and p["rtf"] < 1 else "yellow"
    t.add_row("RTF", Text(f"{p['rtf']:.2f}", style=rtf_style))
    t.add_row("utterances", str(p["n"]))

    # Intent two-pass latency (regression watch): JSON gen vs text gen.
    ip = INTENT_PERF
    if ip["n"]:
        tot = ip["json_ms"] + ip["text_ms"]
        intent_style = "green" if tot < 2500 else ("yellow" if tot < 4000 else "red")
        t.add_row("intent", Text(
            f"JSON {ip['json_ms']:.0f} + txt {ip['text_ms']:.0f} = {tot:.0f}ms",
            style=intent_style))

    # recent notes / events
    notes = Text()
    for level, text, ts in state.notes:
        notes.append(time.strftime("%H:%M:%S", time.localtime(ts)) + " ", style="dim")
        notes.append(text + "\n", style=_LEVEL_STYLE.get(level, ""))

    group = Group(
        _fsm_diagram(state),
        Text(""),                       # spacer
        t,
        Text("\nevents", style="dim"),
        notes,
    )
    return Panel(group, title="State · Perf", title_align="left",
                 border_style="blue")


_RIGHT_MIN = 28   # right panel minimum_size (keep in sync with split_row below)


def render(state, console_height=0, console_width=0):
    """Build the full 3-panel Rich layout for the current state.

    console_height/width let the scrolling panels trim to what fits — by WRAPPED
    rows, so a long message that wraps to several lines is counted correctly and
    the newest text stays visible. 0 → no trimming (show everything).
    """
    # Left column splits chat:intents = 2:1. Each panel spends ~3 rows on
    # border+title, so subtract that from the inner text-row budget.
    if console_height and console_height > 6:
        chat_h = max(1, (console_height * 2) // 3 - 3)
        intent_h = max(1, console_height // 3 - 3)
    else:
        chat_h = intent_h = 0

    # Left column width: total minus the right panel (ratio 1, min _RIGHT_MIN),
    # minus panel borders/padding (~4). Used to estimate text wrapping.
    if console_width and console_width > _RIGHT_MIN + 10:
        right_w = max(_RIGHT_MIN, console_width // 3)
        chat_w = max(10, console_width - right_w - 4)
    else:
        chat_w = 0

    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1, minimum_size=_RIGHT_MIN),
    )
    layout["left"].split_column(
        Layout(_chat_panel(state, chat_h, chat_w), name="chat", ratio=2),
        Layout(_intent_panel(state, intent_h, chat_w), name="intents", ratio=1),
    )
    layout["right"].update(_state_panel(state))
    return layout


# ── live driver ───────────────────────────────────────────────────────────────

class FsttmTUI:
    """Owns the Rich Live and schedules periodic refreshes on the asyncio loop."""

    available = _RICH

    def __init__(self, loop, state, hz=8, log_path=None):
        self.loop = loop
        self.state = state
        self.interval = 1.0 / hz
        self._live = None
        self._stop = False
        # Where stray fd-2 output goes while the alt-screen is up. Default a file
        # next to the cwd so a background traceback / native warning isn't lost.
        self.log_path = log_path or os.path.join(os.getcwd(), "fsttm-tui.log")
        self._saved_fd2 = None
        self._log_fd = None
        self._saved_pystderr = None

    def _redirect_stderr(self):
        # Native libraries (PyAV/ffmpeg avdevice probing X11 → "xcb_…", ALSA,
        # whisper.cpp) write straight to file descriptor 2, bypassing
        # sys.stderr. Those writes corrupt the Rich alt-screen and make it jump.
        # Redirect fd 2 (and Python's sys.stderr) to a log file for the TUI's
        # lifetime; restore on stop so a final crash is still visible.
        try:
            self._log_fd = os.open(self.log_path,
                                   os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._saved_fd2 = os.dup(2)
            sys.stderr.flush()
            os.dup2(self._log_fd, 2)
            # Point Python's stderr at the same file (new object; keep the old
            # to restore). Tracebacks from asyncio callbacks land in the log.
            self._saved_pystderr = sys.stderr
            sys.stderr = os.fdopen(os.dup(2), "w", buffering=1)
        except Exception:
            self._saved_fd2 = None   # best-effort; leave stderr alone on failure

    def _restore_stderr(self):
        try:
            if self._saved_pystderr is not None:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                sys.stderr = self._saved_pystderr
                self._saved_pystderr = None
            if self._saved_fd2 is not None:
                os.dup2(self._saved_fd2, 2)
                os.close(self._saved_fd2)
                self._saved_fd2 = None
            if self._log_fd is not None:
                os.close(self._log_fd)
                self._log_fd = None
        except Exception:
            pass

    def start(self):
        if not _RICH:
            return
        self._redirect_stderr()
        self._console = Console()
        sz = self._console.size
        self._live = Live(render(self.state, sz.height, sz.width),
                          console=self._console, screen=True, auto_refresh=False)
        self._live.start()
        _ACTIVE[0] = True
        self.loop.call_soon(self._tick)

    def _tick(self):
        if self._stop or self._live is None:
            return
        try:
            sz = self._console.size
            self._live.update(
                render(self.state, sz.height, sz.width), refresh=True)
        except Exception:
            pass
        self.loop.call_later(self.interval, self._tick)

    def stop(self):
        self._stop = True
        _ACTIVE[0] = False
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._restore_stderr()
