"""
Async conversation tests for the FSTTM N09-1071 state machine.

These tests validate the FSM event-to-action mapping that the reactive pipeline
uses. They run without any audio hardware, models, or network connections.

Scenarios covered:
  1. Normal turn exchange  (USER → FREEu → SYSTEM → FREEs → USER)
  2. Barge-in, user wins  (SYSTEM → BOTHs → USER)
  3. Barge-in, system keeps (SYSTEM → BOTHs → SYSTEM)
  4. Timeout, system regrabs (SYSTEM → FREEs → SYSTEM)
  5. Cost model structure  (correct keys, W increases in FREEu, G constant)
  6. Async pipeline event queue (end-to-end event mapping via asyncio.Queue)
"""
import asyncio
import time
import pytest
from fsttm.fsttm import Model


# ──────────────────────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────────────────────

class AsyncConversationHarness:
    """
    Wraps a Model instance and records all floor-change callbacks.
    Methods mirror the pipeline events that will drive the FSM.
    """

    def __init__(self):
        self.model = Model()
        self.events = []
        self.model.system_cb = lambda action, had: self.events.append(
            ('sys_floor', action, had)
        )
        self.model.user_cb = lambda action, had: self.events.append(
            ('usr_floor', action, had)
        )

    # ── pipeline event → FSM action ──────────────────────────────────────────

    def vad_speech_started(self):
        """VAD detected start of user speech → user grabs floor."""
        self.model.user_action('G')

    def vad_utterance_done(self):
        """VAD detected end of utterance → user releases floor."""
        self.model.user_action('R')

    def llm_response_ready(self):
        """LLM finished generation, system about to speak → system grabs floor."""
        self.model.system_action('G')

    def tts_playback_done(self):
        """TTS finished speaking → system releases floor."""
        self.model.system_action('R')

    def user_barges_in(self):
        """User speaks while system holds floor → barge-in."""
        self.model.user_action('G')

    def system_yields(self):
        """System detects barge-in and releases floor."""
        self.model.system_action('R')

    def system_regrabs(self):
        """System re-grabs after timeout (user didn't respond)."""
        self.model.system_action('G')

    @property
    def state(self):
        return self.model.state

    def cost(self):
        return self.model.system_actions_cost()


# ──────────────────────────────────────────────────────────────────────────────
# Synchronous scenario tests
# ──────────────────────────────────────────────────────────────────────────────

def test_normal_turn():
    """
    USER → FREEu → SYSTEM → FREEs → USER
    Standard: user speaks, system replies, back to user.
    """
    h = AsyncConversationHarness()
    assert h.state == 'USER'

    h.vad_utterance_done()        # user releases floor after speaking
    assert h.state == 'FREEu'

    h.llm_response_ready()        # system grabs floor to reply
    assert h.state == 'SYSTEM'

    h.tts_playback_done()         # system done → releases floor
    assert h.state == 'FREEs'

    h.vad_speech_started()        # user re-grabs floor
    assert h.state == 'USER'


def test_barge_in_user_wins():
    """
    SYSTEM → BOTHs → USER
    User interrupts system; system yields (detects barge-in, stops speaking).
    """
    h = AsyncConversationHarness()
    h.model.state = 'SYSTEM'
    h.model.system = 'K'  # system is speaking (Keep)

    h.user_barges_in()            # user grabs → overlap
    assert h.state == 'BOTHs'

    h.system_yields()             # system releases → user wins
    assert h.state == 'USER'


def test_barge_in_system_keeps():
    """
    SYSTEM → BOTHs → SYSTEM
    User pokes in but then backs off; system keeps the floor.
    """
    h = AsyncConversationHarness()
    h.model.state = 'SYSTEM'
    h.model.system = 'K'

    h.user_barges_in()            # user grabs → overlap
    assert h.state == 'BOTHs'

    h.model.user_action('R')      # user backs off (releases)
    assert h.state == 'SYSTEM'


def test_timeout_system_regrabs():
    """
    SYSTEM → FREEs → SYSTEM
    System finishes, user stays silent past timeout, system re-asks.
    """
    h = AsyncConversationHarness()
    h.model.state = 'SYSTEM'
    h.model.system = 'K'

    h.tts_playback_done()         # system finishes → FREEs
    assert h.state == 'FREEs'

    h.system_regrabs()            # timeout: system re-grabs
    assert h.state == 'SYSTEM'


def test_cost_model_free_user():
    """
    In FREEu: cost['W'] grows with τ; cost['G'] is constant.
    In SYSTEM: cost has 'K' and 'R' keys.
    In USER: cost has 'W' and 'G' keys.
    """
    h = AsyncConversationHarness()

    # USER state
    c_user = h.cost()
    assert set(c_user.keys()) == {'W', 'G'}

    # FREEu state - wait cost should be near 0 at τ=0, grab cost is constant
    h.model.state = 'FREEu'
    h.model.state_start_time = int(time.time() * 1000)  # reset clock
    c_free = h.cost()
    assert set(c_free.keys()) == {'W', 'G'}
    # G cost is (1 - P_F_pause) * C_u = 0.62 * 5000 = 3100 (fixed)
    # W cost is P_F_pause * C_g_pause * tau ≈ 0 at tau=0
    assert c_free['G'] > c_free['W']  # system should wait initially

    # SYSTEM state
    h.model.state = 'SYSTEM'
    h.model.state_start_time = int(time.time() * 1000)
    c_sys = h.cost()
    assert set(c_sys.keys()) == {'K', 'R'}
    # K cost is P_B * C_o(tau) ≈ 0.1 * exp(0.1) ≈ 0.11 (small, keep)
    # R cost is (1-P_B)*C_s = 0.9 * 100 = 90 (bigger, don't release without reason)
    assert c_sys['K'] < c_sys['R']  # system should keep speaking at τ=0


def test_cost_grows_in_free_state():
    """W cost in FREEu increases with τ; G stays constant."""
    h = AsyncConversationHarness()
    h.model.state = 'FREEu'

    # Sample cost at t=0
    h.model.state_start_time = int(time.time() * 1000)
    c0 = h.cost()

    # Fake a 500ms elapsed pause by backdating state_start_time
    h.model.state_start_time = int(time.time() * 1000) - 500
    c500 = h.cost()

    assert c500['W'] > c0['W']         # wait cost grew
    assert abs(c500['G'] - c0['G']) < 1  # grab cost is constant


# ──────────────────────────────────────────────────────────────────────────────
# Async pipeline simulation test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_pipeline_events():
    """
    Simulate the full async pipeline event flow via asyncio.Queue.
    Mirrors how server.py will feed events to the FSM after wire-up.

    Event sequence: vad_start → vad_stop → stt_result → llm_done → tts_done
    Expected state: USER → USER → FREEu → FREEu → SYSTEM → FREEs
    """
    h = AsyncConversationHarness()
    q: asyncio.Queue = asyncio.Queue()
    states_at: list[tuple] = []

    async def feed():
        for ev in [
            ('vad_start',   None),     # user starts speaking
            ('vad_stop',    None),     # user stops → utterance complete
            ('stt_result',  'hello'),  # STT produced text
            ('llm_done',    'Hi!'),    # LLM finished → system will speak
            ('tts_done',    None),     # TTS finished → floor released
        ]:
            await q.put(ev)
            await asyncio.sleep(0.005)

    async def consume():
        mapping = {
            'vad_start':  lambda _: h.vad_speech_started(),
            'vad_stop':   lambda _: h.vad_utterance_done(),
            'stt_result': lambda _: None,            # no FSM action yet
            'llm_done':   lambda _: h.llm_response_ready(),
            'tts_done':   lambda _: h.tts_playback_done(),
        }
        while True:
            kind, data = await q.get()
            mapping[kind](data)
            states_at.append((kind, h.state))
            q.task_done()
            if kind == 'tts_done':
                break

    await asyncio.gather(feed(), consume())

    state_map = dict(states_at)
    assert state_map['vad_start']  == 'USER'    # user grabbed (self-loop)
    assert state_map['vad_stop']   == 'FREEu'   # user released
    assert state_map['stt_result'] == 'FREEu'   # no change, gap open
    assert state_map['llm_done']   == 'SYSTEM'  # system grabbed
    assert state_map['tts_done']   == 'FREEs'   # system released


@pytest.mark.asyncio
async def test_async_barge_in_sequence():
    """
    Barge-in scenario via event queue.
    System speaking → user barges in → system yields.
    """
    h = AsyncConversationHarness()
    h.model.state = 'SYSTEM'
    h.model.system = 'K'

    q: asyncio.Queue = asyncio.Queue()
    states_at: list[tuple] = []

    async def feed():
        for ev in [
            ('tts_started',    None),  # system started TTS
            ('vad_during_tts', None),  # user spoke during TTS → barge-in
            ('stop_generate',  None),  # pipeline cancels LLM
            ('tts_cancelled',  None),  # TTS subprocess killed
            ('system_yields',  None),  # system releases floor
        ]:
            await q.put(ev)
            await asyncio.sleep(0.005)

    async def consume():
        mapping = {
            'tts_started':    lambda _: None,            # already SYSTEM
            'vad_during_tts': lambda _: h.user_barges_in(),
            'stop_generate':  lambda _: None,            # pipeline side-effect
            'tts_cancelled':  lambda _: None,            # pipeline side-effect
            'system_yields':  lambda _: h.system_yields(),
        }
        while True:
            kind, data = await q.get()
            mapping[kind](data)
            states_at.append((kind, h.state))
            q.task_done()
            if kind == 'system_yields':
                break

    await asyncio.gather(feed(), consume())

    state_map = dict(states_at)
    assert state_map['tts_started']    == 'SYSTEM'  # system holds floor
    assert state_map['vad_during_tts'] == 'BOTHs'   # overlap state
    assert state_map['stop_generate']  == 'BOTHs'   # still overlap
    assert state_map['tts_cancelled']  == 'BOTHs'   # still overlap
    assert state_map['system_yields']  == 'USER'    # user wins floor


if __name__ == '__main__':
    import os, sys
    pytest.main([os.path.abspath(__file__), '-v', '--asyncio-mode=auto'])
