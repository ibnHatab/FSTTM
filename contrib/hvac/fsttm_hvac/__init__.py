"""fsttm-hvac — HVAC / vehicle intent domain for the FSTTM engine.

Provides the `hvac` entry in the `fsttm.domains` registry: climate / lights /
body / manual intent modules, the car-flavored prompt + few-shot examples, the
VHAL command translation (hvac-react PROTOCOL.md), and the dispatcher that
POSTs commands to the hvac-react backend and answers STATUS from live state.
"""
