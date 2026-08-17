"""
HVAC Bridge — forwards FSTTM IntentResult to the hvac-react backend.

On each IntentResult:
  1. translate intent_json via fsttm_hvac.provider.translate
  2. POST each resulting command to <backend_url>/command
  3. Log the delta changes returned by the backend
  4. If the backend is unreachable, warn and continue (never blocks the pipeline)
"""
import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)


class HvacBridge:
    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        # Live state cache: {prop_name: {str(area): value}}. Seeded by
        # refresh_state() and updated from every /command response (which returns
        # the post-change snapshot) so STATUS answers use REAL values, not
        # LLM-invented placeholders. See get_value() / interpolate().
        self._state: dict = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _ingest_props(self, props) -> None:
        if isinstance(props, dict):
            self._state.update(props)

    async def post_intent(self, intent_json: dict) -> None:
        """Translate intent and POST each protocol command to the backend."""
        if not intent_json:
            return
        from fsttm_hvac.provider import translate as intent_to_protocol_cmd
        cmds = intent_to_protocol_cmd(intent_json)
        if not cmds:
            return

        client = await self._get_client()
        for cmd in cmds:
            try:
                resp = await client.post(f"{self.url}/command", json=cmd)
                if resp.status_code == 200:
                    data = resp.json()
                    # The command response carries the full post-change snapshot —
                    # cache it so the next STATUS query has fresh values for free.
                    self._ingest_props(data.get("props"))
                    changes = data.get("changes", [])
                    if changes:
                        log.info("[hvac-bridge] ✓ %s → %s", cmd, changes)
                    else:
                        log.debug("[hvac-bridge] no-op: %s", cmd)
                else:
                    log.warning("[hvac-bridge] %s status=%d: %s",
                                cmd, resp.status_code, resp.text[:120])
            except httpx.ConnectError:
                log.warning("[hvac-bridge] backend unreachable at %s", self.url)
            except Exception as exc:
                log.warning("[hvac-bridge] error posting %s: %s", cmd, exc)

    async def get_state(self) -> Optional[dict]:
        """Fetch full VHAL snapshot from the backend (also refreshes the cache)."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.url}/state")
            if resp.status_code == 200:
                data = resp.json()
                self._ingest_props(data.get("props"))
                return data
        except Exception as exc:
            log.warning("[hvac-bridge] get_state failed: %s", exc)
        return None

    async def refresh_state(self) -> None:
        """Seed/refresh the cache from the backend (call at startup + periodically)."""
        await self.get_state()

    def get_value(self, name: str, area: int = 1):
        """Cached value of a property at an area, or the first area present, or
        None. Synchronous — reads the cache, no network call."""
        areas = self._state.get(name)
        if not isinstance(areas, dict) or not areas:
            return None
        if str(area) in areas:
            return areas[str(area)]
        return next(iter(areas.values()))

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
