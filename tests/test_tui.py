"""
Headless tests for the Rich 3-panel TUI (fsttm/tui.py).

These exercise the pure state + render path — no event loop, no Live, no audio.
Skipped if `rich` is not installed (TUI degrades to stdout there anyway).
"""
import pytest

tui = pytest.importorskip("fsttm.tui")
if not tui.FsttmTUI.available:                       # rich import failed
    pytest.skip("rich not available", allow_module_level=True)

from rich.console import Console


def _render_to_text(state, width=100, height=30):
    con = Console(width=width, height=height)
    with con.capture() as cap:
        con.print(tui.render(state, console_height=height, console_width=width))
    return cap.get()


def test_render_empty_state():
    out = _render_to_text(tui.TUIState())
    # Three panels present, with placeholders for the empty scroll panels.
    assert "Chat" in out
    assert "Intents" in out
    assert "State" in out
    assert "waiting for speech" in out
    assert "no intents yet" in out


def test_chat_and_intent_population():
    st = tui.TUIState()
    st.add_user("make it warmer")
    st.add_assistant("Raising the temperature now.")
    st.add_intent('{"action":"set_temp","value":24}', "Raising to 24.")
    out = _render_to_text(st)
    assert "make it warmer" in out
    assert "set_temp" in out
    assert st.turns == 1


def test_system_intent_in_intent_panel():
    # The attention classifier's decisions show in the intents panel, tagged
    # 'sys:' with the utterance that was classified.
    st = tui.TUIState()
    st.add_system_intent("that's all for today", "sleep")
    st.add_system_intent("what's the weather", "command")
    out = _render_to_text(st)
    assert "sys: sleep" in out
    assert "sys: command" in out
    assert "that's all for today" in out


def test_hvac_and_system_intents_coexist():
    st = tui.TUIState()
    st.add_intent('{"action":"set_temp"}', "ok")
    st.add_system_intent("go to sleep", "sleep")
    out = _render_to_text(st)
    assert "set_temp" in out and "sys: sleep" in out


def test_fsm_and_perf_fields():
    st = tui.TUIState()
    st.set_fsm("SYSTEM", prev="SYSTEM", action="K")
    st.intent_mode = True
    st.hvac_url = "http://127.0.0.1:8000"; st.hvac_ok = True
    tui.record_stt_perf(120.0, 1.5, 0.08)
    out = _render_to_text(st)
    assert "SYSTEM" in out
    assert "intent" in out
    assert "0.08" in out          # RTF
    assert "127.0.0.1" in out


def _diagram_styles(state):
    """Map each FSM node label to the style it renders with."""
    grp = tui._fsm_diagram(state)
    styles = {}
    for line in grp.renderables:
        for span in getattr(line, "spans", []):
            txt = line.plain[span.start:span.end].strip()
            if txt in tui._FSM_NODES:
                styles[txt] = str(span.style)
    return styles


def test_fsm_diagram_has_all_six_nodes():
    out = _render_to_text(tui.TUIState())
    for node in ("FREEs", "SYSTEM", "BOTHs", "USER", "BOTHu", "FREEu"):
        assert node in out, node


def test_fsm_diagram_colors_current_green_prev_yellow():
    st = tui.TUIState()
    st.set_fsm("BOTHs", prev="SYSTEM")
    styles = _diagram_styles(st)
    assert styles["BOTHs"] == "bold green"     # current
    assert styles["SYSTEM"] == "bold blue"     # just left
    # an untouched node stays dim
    assert styles["FREEu"] == "dim"


def test_fsm_diagram_initial_state_user_green():
    st = tui.TUIState()              # defaults to USER, no prev
    styles = _diagram_styles(st)
    assert styles["USER"] == "bold green"
    assert "yellow" not in styles.values()


def test_record_stt_perf_increments_counter():
    before = tui.STT_PERF["n"]
    tui.record_stt_perf(100.0, 1.0, 0.1)
    assert tui.STT_PERF["n"] == before + 1
    assert tui.STT_PERF["rtf"] == 0.1


def test_chat_tail_keeps_newest_visible():
    st = tui.TUIState()
    for i in range(120):
        st.add_user(f"line {i}")
    out = _render_to_text(st, height=30)
    # Newest message must always be visible; the oldest must be cropped.
    assert "line 119" in out
    assert "line 0" not in out


def test_chat_long_wrapped_message_keeps_newest_visible():
    # A single long assistant message wraps to many rows. The newest message
    # must still show even when prior long messages would overflow the panel —
    # this is the bug where trimming-by-entry (not by wrapped rows) hid it.
    st = tui.TUIState()
    long = "word " * 60                       # wraps to several rows
    for i in range(8):
        st.add_assistant(f"msg{i} " + long)
    st.add_user("FINAL-LINE-MARKER")
    out = _render_to_text(st, height=24, width=80)
    assert "FINAL-LINE-MARKER" in out          # newest visible despite overflow
    assert "msg0" not in out                    # oldest cropped


def test_wrapped_rows_counts_wrapping():
    assert tui._wrapped_rows("short", 80) == 1
    assert tui._wrapped_rows("x" * 81, 80) == 2
    assert tui._wrapped_rows("a\nb\nc", 80) == 3
    assert tui._wrapped_rows("x" * 160, 80) == 2


def test_chat_deque_is_memory_bounded():
    st = tui.TUIState()
    for i in range(tui._MAX_CHAT + 50):
        st.add_user(f"x{i}")
    assert len(st.chat) == tui._MAX_CHAT


def test_notes_bounded_and_styled_levels():
    st = tui.TUIState()
    for i in range(tui._MAX_NOTES + 5):
        st.note(f"event {i}", "warn")
    assert len(st.notes) == tui._MAX_NOTES
    # render with a known level should not raise
    st.note("a good thing", "good")
    st.note("a bad thing", "error")
    _render_to_text(st)


def test_intent_mode_label_chat_vs_intent():
    st = tui.TUIState()
    assert "chat" in _render_to_text(st)
    st.intent_mode = True
    assert "intent" in _render_to_text(st)


def test_stderr_redirect_captures_fd2_and_restores(tmp_path):
    """The TUI redirects fd 2 to a log file so native 'xcb_…' / ffmpeg / ALSA
    writes don't corrupt the alt-screen, and restores it cleanly on stop."""
    import os
    log = tmp_path / "fsttm-tui.log"
    t = tui.FsttmTUI(loop=None, state=tui.TUIState(), log_path=str(log))

    saved_fd2 = os.dup(2)
    try:
        t._redirect_stderr()
        # A raw fd-2 write (what native libs do, bypassing sys.stderr).
        os.write(2, b"xcb_connection_has_error leak\n")
        t._restore_stderr()
    finally:
        os.dup2(saved_fd2, 2)
        os.close(saved_fd2)

    assert log.exists()
    assert "xcb_connection_has_error" in log.read_text()
    # fd 2 is usable again after restore.
    os.write(2, b"")
