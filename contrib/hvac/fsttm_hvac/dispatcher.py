"""
HvacDispatcher — executes translated HVAC intents against the hvac-react
backend and answers STATUS from its live VHAL telemetry.

Carries the domain-side behavior that used to live in fsttm/server.py:
  - bridge connect + periodic state polling (was _init_bridge)
  - POST of protocol commands per IntentResult (was _on_intent_result)
  - manual/RAG grounded answers (was _init_retriever + _on_manual_intent),
    triggered by the {"cmd": "manual"} marker in translate() output
  - STATUS spoken answer from real telemetry (was _local_answer)
  - temperature-placeholder interpolation (was _interpolate_placeholders)

Config block (DomainContext.config):
    backend_url: "http://127.0.0.1:8000"   # null/absent → no bridge
    timeout: 2.0
    manual:
        enabled: true
        store: models/taycan-manual.npz
        embed: models/nomic-embed-text-v1.5.Q4_K_M.gguf
        embed_gpu: false
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from fsttm.domain import DispatchOutcome, DomainContext, DomainDispatcher

from fsttm_hvac.bridge import HvacBridge

_log = logging.getLogger("fsttm_hvac.dispatcher")

# A temperature placeholder specifically — interpolated with the real value
# when the backend cache has one. Matches "[current temperature]", "[temp]", …
_TEMP_PLACEHOLDER_RE = re.compile(r'[\[<][^\]>]*temp[^\]>]*[\]>]', re.IGNORECASE)

_POLL_INTERVAL_S = 10.0


class HvacDispatcher(DomainDispatcher):
    def __init__(self, ctx: DomainContext):
        self.ctx = ctx
        cfg = ctx.config or {}
        self._url: Optional[str] = cfg.get("backend_url") or None
        self._timeout: float = float(cfg.get("timeout", 2.0))
        self._manual_cfg: dict = cfg.get("manual") or {}
        self.bridge: Optional[HvacBridge] = None
        self.retriever = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._url:
            self.bridge = HvacBridge(url=self._url, timeout=self._timeout)
            self.ctx.tui_status(self._url, True)
            self.ctx.emit(f"[hvac-bridge] connected to {self._url}", "good")
            # Seed the state cache so the first STATUS query has real values,
            # then refresh periodically to pick up changes made outside the
            # voice path (e.g. the web UI). /command responses update it live.
            if self.ctx.ensure_future is not None:
                self.ctx.ensure_future(self._seed_and_poll())
        self._init_retriever()

    async def _seed_and_poll(self) -> None:
        while self.bridge is not None:
            try:
                await self.bridge.refresh_state()
            except Exception:
                pass
            await asyncio.sleep(_POLL_INTERVAL_S)

    def _init_retriever(self) -> None:
        m = self._manual_cfg
        if not m.get("enabled"):
            self.retriever = None
            return
        store, embed = m.get("store"), m.get("embed")
        if store and embed:
            try:
                from fsttm.rag import Retriever
                embed_gpu = bool(m.get("embed_gpu", False))
                self.retriever = Retriever(store, embed, embed_gpu=embed_gpu)
                self.ctx.emit(f"[manual] RAG ready ({store}, "
                              f"embed={'GPU' if embed_gpu else 'CPU'})", "good")
            except Exception as e:
                self.ctx.emit(f"[manual] RAG unavailable: {e}", "warn")
                self.retriever = None
        else:
            self.ctx.emit("[manual] enabled but store/embed not set", "warn")

    def close(self) -> None:
        bridge, self.bridge = self.bridge, None
        if bridge is not None and self.ctx.ensure_future is not None:
            self.ctx.ensure_future(bridge.close())

    # ── side effects ──────────────────────────────────────────────────────
    def handle(self, intent: dict, commands: list) -> DispatchOutcome:
        """POST backend commands. Manual markers don't map to a device command
        (their narration runs via defer_narration); meta intents translate to
        no commands at all."""
        device_cmds = [c for c in commands
                       if isinstance(c, dict) and c.get("cmd") != "manual"]
        if device_cmds and self.bridge is not None \
                and self.ctx.ensure_future is not None:
            self.ctx.ensure_future(self.bridge.post_intent(intent))
        return DispatchOutcome.PASS

    # ── narration ─────────────────────────────────────────────────────────
    def defer_narration(self, intent: dict, commands: list) -> bool:
        """Manual/RAG intents: retrieve + narrate a grounded answer through
        ctx.narrate_prompt (ManualGenerate → Response/ResponseDone). Returns
        True only if the RAG answer actually dispatched; False means the
        engine must speak a fallback (so we never deadlock with the floor held
        and nothing narrated)."""
        if not any(isinstance(c, dict) and c.get("cmd") == "manual"
                   for c in commands):
            return False
        if self.retriever is None:
            return False
        from fsttm.rag import build_answer_prompt
        ij = intent or {}
        # Prefer the model's `topic`; fall back to the raw utterance (the model
        # often omits topic — "where is the trunk button" → {HOWTO}, no topic).
        query = (ij.get('topic') or '').strip() or \
                (self.ctx.last_user_text() or '').strip()
        if not query:
            return False
        context, hits = self.retriever.context(query)
        if not context:
            self.ctx.emit(f"[manual] no passages for {query!r}", "warn")
            return False
        pages = sorted({h[2].get('page') for h in hits})
        self.ctx.emit(f"[manual] {query!r} → {len(hits)} passages (pp.{pages})",
                      "info")
        prompt = build_answer_prompt(self.ctx.assistant_name, query, context)
        self.ctx.narrate_prompt(prompt)
        return True

    def local_answer(self, intent: dict) -> Optional[str]:
        """STATUS → REAL telemetry from the backend cache (set + current), not
        an LLM-invented value."""
        ij = intent or {}
        if ij.get('intent') != 'STATUS':
            return None
        br = self.bridge
        area = int(ij.get('area') or 1) or 1
        sset = br.get_value('HVAC_TEMPERATURE_SET', area) if br else None
        cur = br.get_value('HVAC_TEMPERATURE_CURRENT', area) if br else None
        zone = {1: "driver side", 4: "passenger side"}.get(area, "")
        where = (" on the " + zone) if zone else ""
        if sset is not None and cur is not None:
            return ("Set to {:g} degrees{}, currently {:g}."
                    .format(float(sset), where, float(cur)))
        val = sset if sset is not None else cur
        if val is not None:
            return "Temperature{} is {:g} degrees.".format(where, float(val))
        return "I don't have the current readings right now."

    def interpolate(self, text: str) -> str:
        """Replace a temperature placeholder with the live reading (just the
        number; the sentence usually already says "degrees")."""
        if not text or ('[' not in text and '<' not in text):
            return text
        temp = (self.bridge.get_value('HVAC_TEMPERATURE_CURRENT')
                if self.bridge else None)
        if temp is None:
            return text
        return _TEMP_PLACEHOLDER_RE.sub("{:g}".format(float(temp)), text)

    def status_line(self):
        return (self._url, self.bridge is not None) if self._url else None
