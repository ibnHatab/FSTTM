"""
Domain plugin seam — the engine's only view of an intent domain.

A *domain* (HVAC car controls, robot-dog commands, …) owns the complete
intent language: the JSON schema the grammar constrains the LLM to, the
system prompt that teaches it, the translation from intent JSON to command
dicts, and the dispatcher that executes those commands against a backend.
The engine never hardcodes intent names, schema fields, or backend wiring —
it talks to the active :class:`DomainProvider` and its
:class:`DomainDispatcher` exclusively.

Domains register through the ``fsttm.domains`` entry-point group::

    [project.entry-points."fsttm.domains"]
    hvac = "fsttm_hvac.provider:PROVIDER"

and a deployment selects ONE via config (``system.domain``). With no domain
configured the engine runs plain chat through :class:`NullProvider`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, runtime_checkable

_log = logging.getLogger("fsttm.domain")


# ── engine meta behaviours ────────────────────────────────────────────────────
# Intents the ENGINE resolves (clock answers, conversational replies, polite
# refusal). A provider maps its own intent JSON onto these names via
# meta_intent(); everything else is domain business.
META_TIME = "TIME"
META_DATE = "DATE"
META_CHITCHAT = "CHITCHAT"
META_UNKNOWN = "UNKNOWN"


class DispatchOutcome(Enum):
    PASS = auto()      # engine narrates (local_answer or the LLM ack)
    DEFERRED = auto()  # dispatcher started its own narration (e.g. RAG answer)


@dataclass
class DomainContext:
    """Engine services handed to a dispatcher at construction time."""
    config: dict                                  # the domain's raw config block
    assistant_name: str = "Nina"
    emit: Callable[[str, str], None] = lambda text, level="info": None
    # Push a fully-formed prompt into the LLM; the streamed answer is narrated
    # through the normal Response/ResponseDone path (context='manual').
    narrate_prompt: Callable[[str], None] = lambda prompt: None
    last_user_text: Callable[[], str] = lambda: ""
    ensure_future: Callable = None                # asyncio.ensure_future or equivalent
    tui_status: Callable[[str, bool], None] = lambda label, ok: None


class DomainDispatcher:
    """Executes translated commands against the domain's backend.

    Default implementation is inert — override what the domain needs. The
    engine calls, in order per intent: ``handle`` (side effects; may DEFER
    narration), then ``local_answer`` / ``interpolate`` when composing the
    spoken reply.
    """

    def start(self) -> None:
        """Connect backends, start polling, load retrievers."""

    def close(self) -> None:
        pass

    def handle(self, intent: dict, commands: list) -> DispatchOutcome:
        """Execute side effects for a translated intent. Return DEFERRED when
        the dispatcher started its own narration (the engine then skips the
        acknowledgment)."""
        return DispatchOutcome.PASS

    def defer_narration(self, intent: dict, commands: list) -> bool:
        """Called when the engine is about to narrate an intent's spoken ack.
        Return True if the dispatcher takes over narration instead (e.g. a
        RAG-grounded answer streamed via ctx.narrate_prompt). False → the
        engine speaks local_answer / the interpolated ack."""
        return False

    def local_answer(self, intent: dict) -> Optional[str]:
        """Deterministic spoken answer for intents the domain resolves from
        live state (e.g. STATUS from backend telemetry). None → use the LLM
        acknowledgment."""
        return None

    def interpolate(self, text: str) -> str:
        """Resolve domain placeholders in a spoken line (e.g. the live
        temperature). Default: identity."""
        return text

    def status_line(self) -> Optional[tuple]:
        """(label, ok) for the TUI domain row, or None."""
        return None


@runtime_checkable
class DomainProvider(Protocol):
    """The complete intent language of one domain. Duck-typed — contrib
    packages implement this shape; ``@runtime_checkable`` only spot-checks
    attribute presence."""
    name: str
    sub_domains: list          # toggleable sub-modules ([] if monolithic)

    def build_schema(self, enabled=None) -> dict: ...
    def build_grammar(self, enabled=None): ...
    def build_prompt(self, enabled=None, variant=None) -> str: ...
    def translate(self, intent: dict, enabled=None) -> list: ...
    def meta_intent(self, intent: dict) -> Optional[str]: ...
    def chitchat_system(self, assistant_name: str) -> Optional[str]: ...
    def make_dispatcher(self, ctx: DomainContext) -> DomainDispatcher: ...


# ── grammar compilation (shared helper) ───────────────────────────────────────
_GRAMMAR_CACHE: dict = {}


def compile_grammar(schema: dict):
    """JSON schema → LlamaGrammar, cached by schema content. llama_cpp prints
    the GBNF to fd 1 while converting; silence it (the TUI owns stdout)."""
    key = json.dumps(schema, sort_keys=True)
    if key not in _GRAMMAR_CACHE:
        from llama_cpp import LlamaGrammar
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(1)
        try:
            os.dup2(devnull, 1)
            _GRAMMAR_CACHE[key] = LlamaGrammar.from_json_schema(json.dumps(schema))
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(devnull)
    return _GRAMMAR_CACHE[key]


# ── null provider (plain chat, no domain installed) ──────────────────────────
class NullProvider:
    """No-domain deployment: no intent grammar, no commands, generic persona."""
    name = "null"
    sub_domains: list = []

    def build_schema(self, enabled=None) -> dict:
        return {"type": "object",
                "properties": {"intent": {"type": "string",
                                          "enum": [META_CHITCHAT, META_UNKNOWN]}},
                "required": ["intent"], "additionalProperties": False}

    def build_grammar(self, enabled=None):
        return compile_grammar(self.build_schema(enabled))

    def build_prompt(self, enabled=None, variant=None) -> str:
        return ("Classify the utterance: a greeting or remark to the assistant "
                'is {"intent":"CHITCHAT"}; anything else is {"intent":"UNKNOWN"}. '
                "Respond with only the JSON.")

    def translate(self, intent, enabled=None) -> list:
        return []

    def meta_intent(self, intent) -> Optional[str]:
        name = (intent or {}).get("intent")
        return name if name in (META_TIME, META_DATE, META_CHITCHAT,
                                META_UNKNOWN) else None

    def chitchat_system(self, assistant_name: str) -> Optional[str]:
        return None

    def make_dispatcher(self, ctx: DomainContext) -> DomainDispatcher:
        return DomainDispatcher()


# ── provider registry ─────────────────────────────────────────────────────────
_ACTIVE: list = [None]      # the deployment's provider (single, config-chosen)


def available_domains() -> dict:
    """Installed entry points in the fsttm.domains group: name → EntryPoint."""
    from importlib.metadata import entry_points
    return {ep.name: ep for ep in entry_points(group="fsttm.domains")}


def load_provider(name: Optional[str]) -> "DomainProvider":
    """Resolve + activate a domain provider by entry-point name.
    name=None → NullProvider (plain chat)."""
    if not name:
        provider = NullProvider()
    else:
        eps = available_domains()
        if name not in eps:
            raise LookupError(
                f"domain {name!r} not installed "
                f"(available: {sorted(eps) or 'none'}) — pip install the "
                f"matching contrib package (e.g. fsttm-{name})")
        provider = eps[name].load()
        _log.info("domain provider loaded: %s (sub-domains: %s)",
                  provider.name, ", ".join(provider.sub_domains) or "-")
    _ACTIVE[0] = provider
    return provider


def active_provider() -> "DomainProvider":
    """The deployment's provider. Before explicit load_provider() (legacy
    configs), fall back to the sole installed domain, preferring 'hvac' for
    back-compat, else NullProvider."""
    if _ACTIVE[0] is None:
        eps = available_domains()
        fallback = "hvac" if "hvac" in eps else (sorted(eps)[0] if eps else None)
        if fallback:
            _log.info("no domain configured — defaulting to %r", fallback)
        load_provider(fallback)
    return _ACTIVE[0]
