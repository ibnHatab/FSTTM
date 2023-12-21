import os
import argparse
import re

from dataclasses import dataclass, field
from typing import List, Dict

# Based on https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp

@dataclass
class GptParams:
    seed: int = -1
    n_threads: int = min(4, os.cpu_count() or 1)
    n_predict: int = 128
    n_ctx: int = 512
    n_batch: int = 8
    n_keep: int = 0

    top_k: int = 40
    top_p: float = 0.95
    tfs_z: float = 1.00
    typical_p: float = 1.00
    temp: float = 0.80
    repeat_penalty: float = 1.10
    repeat_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    model: str = "./models/llama-7B/ggml-model.bin"
    prompt: str = ""
    path_session: str = ""
    input_prefix: str = " "
    input_suffix: str = ""
    antiprompt: List[str] = field(default_factory=list)

    random_prompt: bool = False
    use_color: bool = False
    penalize_nl: bool = True
    verbose_prompt: bool = False
    input_echo: bool = True,


