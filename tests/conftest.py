import os
import sys


def pytest_configure(config):
    sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__)) + "/..")

