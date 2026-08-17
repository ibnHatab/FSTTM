"""fsttm-dog — robot-dog intent domain for the FSTTM engine.

Provides the `dog` entry in the `fsttm.domains` registry: a GOAT-style
constrained intent language (LOCAL_ACTION / QUERY / NAVIGATE / FIND / FOLLOW /
STOP / CANCEL) with semantic targets and spatial relations. The LLM produces
intents; semantic perception resolves them into spatial goals; classical
navigation (nav2) executes them. See spec.md.
"""
