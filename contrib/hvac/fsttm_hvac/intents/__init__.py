"""HVAC intent modules — importing this package registers all built-in
domains (climate, lights, body, manual) into fsttm_hvac.registry."""
from fsttm_hvac.intents import climate, lights, body, manual  # noqa: F401
