"""
PipeWire/PulseAudio echo cancellation via module-echo-cancel.

Loads the module on enter, unloads on exit.  The two virtual devices created:
  fsttm_ec_source  — mic with TTS echo removed  → PyAudio capture
  fsttm_ec_sink    — virtual sink; TTS plays here so the AEC has the reference

Optionally chains RNNoise (LADSPA) on top of the AEC output for extra noise
suppression — enable via EchoCancelSession(rnnoise=True) / config aec.rnnoise.
"""
from __future__ import annotations  # PEP 563: lazy annotations for Python 3.8
import os
import subprocess

# Fallback master source if auto-detection finds no Jabra. Kept for back-compat;
# overridden by detect_jabra_source() / an explicit source_master.
JABRA_SOURCE = (
    "alsa_input.usb-0b0e_Jabra_Link_370_70BF928749C2-00.mono-fallback"
)
EC_SOURCE  = "fsttm_ec_source"
EC_SINK    = "fsttm_ec_sink"
RNN_SOURCE = "fsttm_rnnoise_source"   # AEC + RNNoise chain output


def detect_jabra_source(fallback: str = JABRA_SOURCE) -> str:
    """Return the first PulseAudio input source containing 'Jabra'.

    Hardware varies (Link 370 dongle, EVOLVE 20 headset, …), so match by name
    rather than hardcoding one USB id. Falls back to `fallback` if none found.
    """
    try:
        r = subprocess.run(['pactl', 'list', 'short', 'sources'],
                           capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            name = line.split('\t')[1] if '\t' in line else ''
            if 'Jabra' in name and name.startswith('alsa_input'):
                return name
    except Exception:
        pass
    return fallback


def _load_rnnoise_chain(source_master: str) -> str | None:
    """
    Chain RNNoise on top of source_master via module-ladspa-source (PA compat).
    Uses the system-installed librnnoise_ladspa.so. Returns the pactl module ID,
    or None if the plugin isn't available.
    """
    if not os.path.exists("/usr/lib/ladspa/librnnoise_ladspa.so"):
        return None
    r = subprocess.run(
        ["pactl", "load-module", "module-ladspa-source",
         f"source_name={RNN_SOURCE}",
         f"master={source_master}",
         "plugin=librnnoise_ladspa",
         "label=noise_suppressor_stereo"],  # stereo matches the 2ch AEC source
        capture_output=True, text=True,
    )
    mod_id = r.stdout.strip()
    return mod_id if r.returncode == 0 and mod_id else None


class EchoCancelSession:
    def __init__(self, source_master: str | None = None,
                 enabled: bool = True, rnnoise: bool = False,
                 method: str = "auto", sink_master: str | None = None):
        # None → auto-detect the connected Jabra at enable() time.
        self._source_master = source_master or detect_jabra_source()
        self._sink_master    = sink_master   # None → PA default sink (module picks)
        self._enabled        = enabled
        self._rnnoise        = rnnoise
        self._method         = (method or "auto").lower()
        self._module_id: str | None = None
        self._rnn_module_id: str | None = None
        self._prev_source: str | None = None
        self._prev_sink: str | None = None

    def _cleanup_stale(self) -> None:
        """Unload any leftover fsttm_ec / rnnoise modules from a crashed session."""
        r = subprocess.run(['pactl', 'list', 'modules', 'short'],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if (f'source_name={EC_SOURCE}' in line
                    or f'sink_name={EC_SINK}' in line
                    or f'source_name={RNN_SOURCE}' in line):
                mod_id = line.split('\t')[0].strip()
                subprocess.run(['pactl', 'unload-module', mod_id], check=False)
                print(f"AEC: cleaned up stale module {mod_id}")

    def enable(self) -> None:
        if not self._enabled:
            print("AEC disabled (config aec.enabled=false)")
            return
        self._cleanup_stale()

        def _base(method):
            cmd = [
                "pactl", "load-module", "module-echo-cancel",
                f"aec_method={method}",
                f"source_name={EC_SOURCE}",
                f"sink_name={EC_SINK}",
                f"source_master={self._source_master}",
            ]
            if self._sink_master:
                cmd.append(f"sink_master={self._sink_master}")
            return cmd

        # webrtc gives AEC + built-in noise suppression; speex is echo-only but
        # loads reliably where webrtc's init fails (some Jetson PA builds). Try in
        # order of preference; "auto" walks the whole list, an explicit method
        # tries only its variants.
        WEBRTC_TUNED = _base("webrtc") + [
            "aec_args=analog_gain_control=0 digital_gain_control=1 "
            "noise_suppression=1 voice_detection=1",
        ]
        attempts = []
        if self._method in ("auto", "webrtc"):
            attempts += [("webrtc+ns", WEBRTC_TUNED), ("webrtc", _base("webrtc"))]
        if self._method in ("auto", "speex"):
            attempts += [("speex", _base("speex"))]

        last_err = ""
        for label, cmd in attempts:
            result = subprocess.run(cmd, capture_output=True, text=True)
            mod = result.stdout.strip()
            if result.returncode == 0 and mod:
                self._module_id = mod
                self._method_used = label
                print(f"AEC enabled [{label}] (module {mod}): "
                      f"{EC_SOURCE} / {EC_SINK}")
                self._maybe_chain_rnnoise()
                self._route_default()
                return
            last_err = result.stderr.strip() or f"{label} init failed"

        raise RuntimeError(
            f"AEC: module-echo-cancel failed (method={self._method}) for master "
            f"{self._source_master!r}: {last_err}")

    def _maybe_chain_rnnoise(self) -> None:
        """Chain RNNoise on top of the AEC output when requested. Sets the
        routed source so _route_default() makes it the PulseAudio default."""
        self._active_source = EC_SOURCE
        if not self._rnnoise:
            return
        rnn_id = _load_rnnoise_chain(EC_SOURCE)
        if rnn_id:
            self._rnn_module_id = rnn_id
            self._active_source = RNN_SOURCE
            import time
            time.sleep(0.3)   # let PipeWire register the node
            print(f"  RNNoise chained: {EC_SOURCE} → {RNN_SOURCE}")
        else:
            print("  RNNoise: /usr/lib/ladspa/librnnoise_ladspa.so not found, skipping")

    def _route_default(self) -> None:
        """Make the (RNNoise or EC) source and EC sink the PulseAudio defaults.

        PyAudio cannot address `fsttm_ec_source` by name (it only exposes raw
        ALSA hw devices plus the generic `pulse`/`default` devices). Routing the
        defaults to the EC source/sink lets capture via the `pulse` device pick
        up the echo-cancelled mic, and playback via `pulse` reach the EC sink.
        """
        src = getattr(self, '_active_source', EC_SOURCE)
        # Remember current defaults so we can restore them on disable().
        info = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
        for line in info.stdout.splitlines():
            if line.startswith('Default Source:'):
                self._prev_source = line.split(':', 1)[1].strip()
            elif line.startswith('Default Sink:'):
                self._prev_sink = line.split(':', 1)[1].strip()
        subprocess.run(['pactl', 'set-default-source', src], check=False)
        subprocess.run(['pactl', 'set-default-sink', EC_SINK], check=False)
        print(f"AEC: default source→{src}, sink→{EC_SINK}")

    def disable(self) -> None:
        if not self._enabled:
            return
        # Restore previous defaults before unloading the modules.
        if getattr(self, '_prev_source', None):
            subprocess.run(['pactl', 'set-default-source', self._prev_source],
                           check=False)
        if getattr(self, '_prev_sink', None):
            subprocess.run(['pactl', 'set-default-sink', self._prev_sink],
                           check=False)
        if self._rnn_module_id:
            subprocess.run(["pactl", "unload-module", self._rnn_module_id],
                           check=False)
            self._rnn_module_id = None
        if self._module_id:
            subprocess.run(["pactl", "unload-module", self._module_id],
                           check=False)
            print(f"AEC disabled (module {self._module_id})")
            self._module_id = None

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *_):
        self.disable()
