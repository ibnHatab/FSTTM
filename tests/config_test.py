
import os
import sys
import pytest

from gpt_fsttm_server.config import parse_config


def test_config():
    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE, '../config.sample.yaml')) as f:
        config_data = f.read()

    config = parse_config(config_data)
    print(config)
