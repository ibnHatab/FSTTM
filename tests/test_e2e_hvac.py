"""
End-to-end HVAC tests: intent pipeline → HvacBridge → FastAPI backend.

Coverage:
  - Backend health + initial state
  - HvacBridge REST tests (set_temperature, fan_up, warmer_both, ac_toggle, unknown)
  - grammar.intent_to_protocol_cmd → backend state mutation

The browser/Playwright UI assertions were removed when the frontend was
replaced with the new cockpit design (the old SIMPLE/EXPERT screens and their
DOM hooks no longer exist). These tests still exercise the full
intent → protocol-command → backend-state path, which is the integration the
FSTTM pipeline depends on.

Run:
    PYTHONPATH=contrib/hvac/hvac-react/backend pytest tests/test_e2e_hvac.py -v --asyncio-mode=auto
"""
import subprocess
import sys
import time
import pytest
import httpx

BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_DIR = "contrib/hvac/hvac-react/backend"

# ── wait helper ──────────────────────────────────────────────────────────────

def _wait_port(url: str, seconds: int = 15) -> bool:
    for _ in range(seconds * 2):
        try:
            if httpx.get(url, timeout=0.5).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hvac_backend():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=BACKEND_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if not _wait_port(BACKEND_URL):
        proc.kill()
        pytest.fail("hvac backend did not start")
    yield proc
    proc.terminate(); proc.wait(timeout=5)


# ── helpers ──────────────────────────────────────────────────────────────────

def post_cmd(cmd: dict) -> dict:
    r = httpx.post(f"{BACKEND_URL}/command", json=cmd, timeout=3.0)
    r.raise_for_status()
    return r.json()


def set_prop(name: str, area: int, value) -> dict:
    return post_cmd({"cmd": "set", "name": name, "area": area, "value": value})


def do_action(action_name: str, **args) -> dict:
    cmd: dict = {"cmd": "action", "action": action_name}
    if args:
        cmd["args"] = args
    return post_cmd(cmd)


def state_prop(name: str, area_str: str):
    s = httpx.get(f"{BACKEND_URL}/state", timeout=2.0).json()
    return s["props"][name][str(area_str)]


# ── Backend / REST tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backend_health(hvac_backend):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BACKEND_URL}/")
    assert r.status_code == 200
    assert "HVAC" in r.json()["service"]


@pytest.mark.asyncio
async def test_initial_state(hvac_backend):
    async with httpx.AsyncClient() as c:
        state = (await c.get(f"{BACKEND_URL}/state")).json()["props"]
    assert "HVAC_TEMPERATURE_SET" in state
    assert "HVAC_FAN_SPEED" in state
    assert state["HVAC_POWER_ON"]["0"] is False


# ── HvacBridge intent pipeline ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bridge_set_temperature(hvac_backend):
    from fsttm.hvac_bridge import HvacBridge
    b = HvacBridge(url=BACKEND_URL)
    await b.post_intent({"intent": "SET_TEMPERATURE", "area": 1, "temp": 21.0})
    assert state_prop("HVAC_TEMPERATURE_SET", "1") == 21.0
    await b.close()


@pytest.mark.asyncio
async def test_bridge_fan_up(hvac_backend):
    from fsttm.hvac_bridge import HvacBridge
    b = HvacBridge(url=BACKEND_URL)
    before = state_prop("HVAC_FAN_SPEED", "1")
    await b.post_intent({"intent": "FAN_UP", "area": 1, "delta": 1})
    assert state_prop("HVAC_FAN_SPEED", "1") == min(7, before + 1)
    await b.close()


@pytest.mark.asyncio
async def test_bridge_warmer_both_zones(hvac_backend):
    from fsttm.hvac_bridge import HvacBridge
    b = HvacBridge(url=BACKEND_URL)
    # reset to known value first
    set_prop("HVAC_TEMPERATURE_SET", 0, 22.0)
    before_l = state_prop("HVAC_TEMPERATURE_SET", "1")
    before_r = state_prop("HVAC_TEMPERATURE_SET", "4")
    await b.post_intent({"intent": "WARMER", "area": 0, "delta": 2})
    assert state_prop("HVAC_TEMPERATURE_SET", "1") == pytest.approx(min(28.0, before_l + 1.0), abs=0.1)
    assert state_prop("HVAC_TEMPERATURE_SET", "4") == pytest.approx(min(28.0, before_r + 1.0), abs=0.1)
    await b.close()


@pytest.mark.asyncio
async def test_bridge_ac_toggle(hvac_backend):
    from fsttm.hvac_bridge import HvacBridge
    b = HvacBridge(url=BACKEND_URL)
    before = state_prop("HVAC_AC_ON", "0")
    await b.post_intent({"intent": "AC_ON", "area": 0})
    assert state_prop("HVAC_AC_ON", "0") != before
    await b.post_intent({"intent": "AC_OFF", "area": 0})
    assert state_prop("HVAC_AC_ON", "0") == before
    await b.close()


@pytest.mark.asyncio
async def test_bridge_unknown_does_nothing(hvac_backend):
    from fsttm.hvac_bridge import HvacBridge
    b = HvacBridge(url=BACKEND_URL)
    s0 = httpx.get(f"{BACKEND_URL}/state").json()["props"]
    await b.post_intent({"intent": "UNKNOWN", "area": 0})
    s1 = httpx.get(f"{BACKEND_URL}/state").json()["props"]
    assert s0 == s1
    await b.close()


# ── grammar → protocol command → backend ─────────────────────────────────────

def test_intent_to_protocol_set_temperature(hvac_backend):
    """SET_TEMPERATURE intent translates to a protocol command that mutates state."""
    from fsttm.grammar import intent_to_protocol_cmd
    set_prop("HVAC_TEMPERATURE_SET", 0, 20.0)
    for cmd in intent_to_protocol_cmd({"intent": "SET_TEMPERATURE", "area": 1, "temp": 25.5}):
        post_cmd(cmd)
    assert state_prop("HVAC_TEMPERATURE_SET", "1") == pytest.approx(25.5, abs=0.1)


def test_intent_to_protocol_ac_on(hvac_backend):
    """AC_ON intent translates to a protocol command that enables A/C."""
    from fsttm.grammar import intent_to_protocol_cmd
    set_prop("HVAC_AC_ON", 0, False)
    for cmd in intent_to_protocol_cmd({"intent": "AC_ON", "area": 0}):
        post_cmd(cmd)
    assert state_prop("HVAC_AC_ON", "0") is True
