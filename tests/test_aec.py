"""
Automated AEC (Acoustic Echo Cancellation) integration test.

Verifies the full PipeWire echo-cancel setup:
1. Module loads / unloads cleanly
2. Virtual devices appear after enable, disappear after disable
3. Recording from fsttm_ec_source while playing through fsttm_ec_sink
   yields near-silent output (RMS < 200 on a 16-bit scale of 32767)

Run without hardware interaction — just audio routing machinery.
Skipped automatically when PipeWire/pactl is unavailable.
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8
import subprocess
import time
import wave
import numpy as np
import pytest

from fsttm.aec import EchoCancelSession, EC_SOURCE, EC_SINK


def _pactl_sources() -> list[str]:
    r = subprocess.run(['pactl', 'list', 'sources', 'short'],
                       capture_output=True, text=True)
    return r.stdout


def _pactl_sinks() -> list[str]:
    r = subprocess.run(['pactl', 'list', 'sinks', 'short'],
                       capture_output=True, text=True)
    return r.stdout


def _pipewire_available() -> bool:
    r = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
    return r.returncode == 0 and 'PipeWire' in r.stdout


skip_no_pw = pytest.mark.skipif(
    not _pipewire_available(),
    reason="PipeWire not available"
)


@skip_no_pw
def test_aec_enable_creates_devices():
    """Virtual source and sink appear after enable."""
    assert EC_SOURCE not in _pactl_sources(), "AEC source should not exist before enable"
    assert EC_SINK   not in _pactl_sinks(),   "AEC sink should not exist before enable"

    session = EchoCancelSession()
    session.enable()
    try:
        assert EC_SOURCE in _pactl_sources(), "AEC source missing after enable"
        assert EC_SINK   in _pactl_sinks(),   "AEC sink missing after enable"
    finally:
        session.disable()


@skip_no_pw
def test_aec_disable_removes_devices():
    """Virtual devices disappear after disable."""
    session = EchoCancelSession()
    session.enable()
    session.disable()

    assert EC_SOURCE not in _pactl_sources(), "AEC source still present after disable"
    assert EC_SINK   not in _pactl_sinks(),   "AEC sink still present after disable"


@skip_no_pw
def test_aec_context_manager():
    """EchoCancelSession works as a context manager."""
    with EchoCancelSession():
        assert EC_SOURCE in _pactl_sources()
        assert EC_SINK   in _pactl_sinks()
    assert EC_SOURCE not in _pactl_sources()
    assert EC_SINK   not in _pactl_sinks()


@skip_no_pw
def test_aec_double_disable_safe():
    """Calling disable twice does not raise."""
    session = EchoCancelSession()
    session.enable()
    session.disable()
    session.disable()   # should not raise


@skip_no_pw
def test_aec_echo_suppression():
    """
    Play a 440 Hz sine through fsttm_ec_sink and simultaneously record from
    fsttm_ec_source.  The recorded signal should have very low RMS — PipeWire's
    WebRTC AEC must have removed the playback from the mic signal.

    Acceptance threshold: RMS < 200  (full-scale 16-bit = 32767)
    """
    # Generate 2 s of 440 Hz sine at 22050 Hz, 16-bit mono
    rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 28000).astype(np.int16)
    sine_bytes = sine.tobytes()

    with EchoCancelSession():
        time.sleep(0.4)   # let PipeWire settle the routing graph

        rec = subprocess.Popen(
            ['arecord', '-D', f'pulse:{EC_SOURCE}',
             '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', '2',
             '/tmp/aec_echo_test.wav'],
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.05)

        # Play sine via aplay (stdin pipe, terminates naturally after 2 s)
        play = subprocess.Popen(
            ['aplay', '-D', f'pulse:{EC_SINK}',
             '-f', 'S16_LE', '-r', str(rate), '-c', '1', '-q'],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        play.communicate(sine_bytes)

        rec.wait(timeout=6)

    with wave.open('/tmp/aec_echo_test.wav') as wf:
        samples = np.frombuffer(
            wf.readframes(wf.getnframes()), dtype=np.int16
        ).astype(np.float32)

    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    print(f"\nAEC echo test — RMS after cancellation: {rms:.1f} (threshold <200)")
    assert rms < 200, (
        f"Echo not suppressed: RMS={rms:.1f} (expected <200). "
        "Check PipeWire AEC config or Jabra device routing."
    )
