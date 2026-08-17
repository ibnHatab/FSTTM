import os
import yaml
import pytest
from fsttm.config import Config


def test_config():
    """config.sample.yaml parses into a valid Config with all required fields."""
    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE, '../config.sample.yaml')) as f:
        data = yaml.safe_load(f)

    config = Config(**data)

    assert config.vad.rate == 16000
    assert config.vad.device_name == "pulse"
    assert config.stt.model == "models/ggml-base.en-q5_1.bin"
    assert config.tts.backend == "piper"
    assert config.tts.piper["sample_rate"] == 22050
    assert config.tts.piper["model"].endswith('.onnx')
    assert config.gpt.model.endswith('.gguf')
    assert config.gpt.n_ctx == 4096
    assert config.gpt.temp == 0.7
    print(f"\nConfig OK: model={os.path.basename(config.gpt.model)}, "
          f"vad_device={config.vad.device_name}")
