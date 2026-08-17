import yaml
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import reactivex.operators as ops
import cyclotron_std.argparse as argparse


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
    model: str          # path to piper .onnx voice model
    sample_rate: int = 22050
    # Output routing (resolved by name, like aplay -l / arecord -l). PCM is
    # written straight to a PyAudio stream — no aplay subprocess.
    device: Optional[str] = None   # PyAudio output device name substring;
                                   # null → "pulse" (PulseAudio handles convert)
    sink: Optional[str] = None     # PulseAudio sink name substring; null →
    cuda: bool = False             # run ONNX synthesis on GPU (onnxruntime CUDA EP; needs onnxruntime-gpu + cuDNN)
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


class HvacBackend(BaseModel):
    """Optional HTTP backend to forward intent commands to (hvac-react)."""
    url: Optional[str] = None        # e.g. "http://127.0.0.1:8000" — null disables
    timeout: float = 2.0             # per-command POST timeout in seconds


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
    # HVAC / vehicle intent mode — two-pass grammar-constrained generation.
    # Replaces the old gpt.intent_mode / gpt.intent_prompt.
    hvac_intent: bool = False
    # Which intent domains are active (assembled into the grammar + prompt).
    # null → all registered domains; or a subset e.g. ["climate", "lights"].
    intent_domains: Optional[List[str]] = None
    hvac_prompt: Optional[str] = None   # optional EXTRA prompt file appended to
                                        # the auto-assembled domain prompt
    # Intent prompt variant: "one-shot" (lean, fastest), "few-shot" (production,
    # +accuracy), or "few-shot-extra" (max coverage). See intents.PROMPT_VARIANTS.
    prompt_variant: str = "few-shot"
    # Manual RAG — answer how-to/where-is/explain questions (the `manual` intent
    # domain) from a manual. Enable with manual: true AND both paths set.
    manual: bool = False                # master toggle for manual RAG
    manual_store: Optional[str] = None  # path to an ingested .npz vector store
    manual_embed: Optional[str] = None  # path to the embedding GGUF
    manual_embed_gpu: bool = False      # offload the embedder to GPU. Default CPU
                                        # — on a shared-memory Jetson a GPU embedder
                                        # competes with the LLM for VRAM and OOMs.
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
    hvac_backend: HvacBackend = HvacBackend()   # optional — defaults to disabled
    system: System = System()                   # attention / system-intent flags
    aec: AecConfig = AecConfig()                 # echo-cancel / rnnoise


def parse_config(config_data):
    config = config_data.pipe(
        ops.filter(lambda i: i.id == "config"),
        ops.flat_map(lambda i: i.data),
        ops.map(lambda i: yaml.load(i, Loader=yaml.FullLoader)),
        ops.map(lambda i: Config(**i)),
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
