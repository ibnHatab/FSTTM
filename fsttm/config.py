import logging
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import reactivex.operators as ops
import cyclotron_std.argparse as argparse

_log = logging.getLogger("fsttm.config")


class VAD(BaseModel):
    model_config = ConfigDict(extra='allow')
    vad_aggressiveness: int = 3
    device: Optional[int] = None
    device_name: Optional[str] = None   # PipeWire source name (overrides device index)
    rate: int = 16000
    padding_ms: int = 700   # silence duration before closing utterance (300=fast, 700=natural)
    soft_duck: bool = True   # route playback-period speech to a barge-in sentinel
                             # instead of STT; set false to rely on AEC alone and
                             # detect barge-in from a plain VAD onset


class STT(BaseModel):
    model: str = "base"
    language: str = "en"   # force Whisper language; prevents wrong-script hallucinations
    # Parasite phrases — whole-string whisper hallucinations to drop as noise
    # (in addition to []/()/** sound-annotation auto-detection). Null = defaults.
    parasites: Optional[List[str]] = None


class TTS(BaseModel):
    model_config = ConfigDict(extra='allow')
    # Which synth backend to use (fsttm.tts_backends entry-point name). The
    # backend's own settings live in a same-named nested block (tts.piper /
    # tts.rhvoice), passed to SynthBackend.load() verbatim.
    backend: str = "piper"
    piper: Optional[dict] = None    # {model: <onnx>, sample_rate: 22050, cuda: false}
    rhvoice: Optional[dict] = None  # {voice: SLT, rate: 0.3, volume: -0.1}
    # Output routing (resolved by name, like aplay -l / arecord -l). PCM is
    # written straight to a PyAudio stream — no aplay subprocess. Shared by
    # all backends.
    device: Optional[str] = None   # PyAudio output device name substring;
                                   # null → "pulse" (PulseAudio handles convert)
    sink: Optional[str] = None     # PulseAudio sink name substring; null →
                                   # fsttm_ec_sink when AEC is on (so the AEC sees
                                   # the TTS), else the default sink. e.g. "Jabra"


class HttpServer(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    request_max_size: int = 1048576


class Server(BaseModel):
    http: HttpServer


class LogLevel(BaseModel):
    logger: str
    level: str


class Log(BaseModel):
    level: List[LogLevel]


class AecConfig(BaseModel):
    """PulseAudio echo-cancel + optional RNNoise chain."""
    enabled: bool = True   # load module-echo-cancel (AEC + noise suppression)
    rnnoise: bool = False  # chain RNNoise LADSPA on top of the AEC
    # AEC backend: "auto" tries webrtc (AEC+noise-suppression) then falls back to
    # speex (echo-only). On some Jetson PulseAudio builds webrtc fails to init —
    # "auto" lands on speex there. Force a method by name if needed.
    method: str = "auto"   # "auto" | "webrtc" | "speex"


class GptParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    n_ctx: int = 2048
    seed: float = 42
    temp: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    repeat_last_n: int = 256
    n_batch: int = 512
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    model: str
    n_threads: int = 6
    n_predict: int = 512
    safeword: str = ""


class System(BaseModel):
    """System-level behaviour flags (intents, attention / wake word).
    A general place to gate built-in assistant behaviours, extensible as more
    intents are added."""
    model_config = ConfigDict(extra='allow')
    name: str = "Nina"                  # the assistant's name (wake word + prompt)
    # Intent mode — two-pass grammar-constrained generation against the active
    # domain provider's schema. (was: hvac_intent)
    intent_mode: bool = False
    # Which domain provider to load (fsttm.domains entry-point name, e.g.
    # "hvac", "dog"). null → auto: the sole installed domain (hvac preferred),
    # or plain chat when no contrib package is installed.
    domain: Optional[str] = None
    # Which sub-domains of the provider are active (assembled into the grammar
    # + prompt). null → all; or a subset e.g. ["climate", "lights"].
    intent_domains: Optional[List[str]] = None
    intent_prompt: Optional[str] = None  # optional EXTRA prompt file appended
                                         # to the assembled domain prompt
                                         # (was: hvac_prompt)
    # Intent prompt variant: "one-shot" (lean, fastest), "few-shot" (production,
    # +accuracy), or "few-shot-extra" (max coverage).
    prompt_variant: str = "few-shot"
    attention: bool = False             # wake-word layer; start ASLEEP when true.
                                        # Once woken it stays AWAKE unless
                                        # sleep_intent re-enables sleeping.
    sleep_intent: bool = False          # LLM system-intent grammar decides when
                                        # to go back to sleep — needs attention.
                                        # Off → never re-sleeps (always awake).
    wake_words: List[str] = ["nina", "hey nina", "hi nina"]


class Config(BaseModel):
    model_config = ConfigDict(extra='allow')
    vad: VAD
    stt: STT
    tts: TTS
    gpt: GptParams
    server: Server
    log: Log
    system: System = System()                   # attention / system-intent flags
    aec: AecConfig = AecConfig()                # echo-cancel / rnnoise
    # Per-domain config blocks, passed through RAW to the matching provider's
    # dispatcher (the provider validates its own block). E.g.
    #   domains:
    #     hvac: {backend_url: "http://127.0.0.1:8000", timeout: 2.0,
    #            manual: {enabled: true, store: ..., embed: ...}}
    domains: Dict[str, dict] = {}
    # Utterance-level voice filter ("only my voice") — see fsttm.voicefilter.
    #   voice_filter: {enabled: true, provider: speaker, model: <onnx>,
    #                  profiles: <npz>, threshold: 0.45, min_utterance_s: 0.5,
    #                  mode: enforce}       # enforce | shadow
    voice_filter: dict = {}


# Legacy-key mapping (pre-0.2 configs). Applied before validation with a
# deprecation warning per hit, so out-of-tree configs keep booting.
def normalize_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return raw
    raw = dict(raw)
    sysc = dict(raw.get('system') or {})
    domains = dict(raw.get('domains') or {})
    hvac = dict(domains.get('hvac') or {})
    legacy_hvac = False

    def _warn(old, new):
        _log.warning("config: legacy key %s — use %s", old, new)

    if 'hvac_intent' in sysc:
        _warn('system.hvac_intent', 'system.intent_mode')
        sysc.setdefault('intent_mode', sysc.pop('hvac_intent'))
    if 'hvac_prompt' in sysc:
        _warn('system.hvac_prompt', 'system.intent_prompt')
        sysc.setdefault('intent_prompt', sysc.pop('hvac_prompt'))
        legacy_hvac = True
    if 'hvac_backend' in raw:
        _warn('hvac_backend', 'domains.hvac.{backend_url,timeout}')
        hb = raw.pop('hvac_backend') or {}
        if hb.get('url'):
            hvac.setdefault('backend_url', hb.get('url'))
            hvac.setdefault('timeout', hb.get('timeout', 2.0))
            legacy_hvac = True
    if any(k in sysc for k in ('manual', 'manual_store', 'manual_embed',
                               'manual_embed_gpu')):
        _warn('system.manual*', 'domains.hvac.manual.*')
        hvac.setdefault('manual', {
            'enabled': bool(sysc.pop('manual', False)),
            'store': sysc.pop('manual_store', None),
            'embed': sysc.pop('manual_embed', None),
            'embed_gpu': bool(sysc.pop('manual_embed_gpu', False)),
        })
        legacy_hvac = True
    if legacy_hvac and not sysc.get('domain'):
        sysc['domain'] = 'hvac'
    # Flat tts keys (pre-backend schema) → the piper block.
    ttsc = dict(raw.get('tts') or {})
    if 'model' in ttsc and 'piper' not in ttsc:
        _warn('tts.{model,sample_rate,cuda}', 'tts.piper.{model,sample_rate,cuda}')
        ttsc['piper'] = {'model': ttsc.pop('model'),
                         'sample_rate': ttsc.pop('sample_rate', 22050),
                         'cuda': bool(ttsc.pop('cuda', False))}
        ttsc.setdefault('backend', 'piper')
        raw['tts'] = ttsc
    if hvac:
        domains['hvac'] = hvac
    if domains:
        raw['domains'] = domains
    raw['system'] = sysc
    return raw


def parse_config(config_data):
    config = config_data.pipe(
        ops.filter(lambda i: i.id == "config"),
        ops.flat_map(lambda i: i.data),
        ops.map(lambda i: yaml.load(i, Loader=yaml.FullLoader)),
        ops.map(lambda i: Config(**normalize_config(i))),
        ops.share(),
    )
    return config


def parse_arguments(argv):
    parser = argparse.ArgumentParser("Finite-State Turn-Taking Machine")
    parser.add_argument('--config', required=True,
                        help="Path to the server configuration file")
    return argv.pipe(
        ops.skip(1),
        argparse.parse(parser),
    )
