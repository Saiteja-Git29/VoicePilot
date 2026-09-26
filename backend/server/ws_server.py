from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Set

import websockets

from ..logging_setup import log_event
from ..voice.events import ConnectionState
from ..voice.session_manager import SessionManager

logger = logging.getLogger("voicepilot.server.ws")


class StatusServer:
    """Broadcasts SessionManager state to connected UI clients over a plain
    WebSocket. Read-only status channel for this phase: the UI observes
    connection/mic/speaking/interruption/error state, nothing more. No control
    messages are accepted from clients yet — that would be a later phase."""

    def __init__(self, session_manager: SessionManager, host: str, port: int) -> None:
        self._session_manager = session_manager
        self._host = host
        self._port = port
        self._clients: Set[object] = set()
        self._server = None
        session_manager.on_state_change(self._on_state_change)

    def _on_state_change(self, state: ConnectionState, message: Optional[str]) -> None:
        payload = json.dumps({"type": "state", "state": state.value, "message": message})
        for client in list(self._clients):
            asyncio.create_task(self._safe_send(client, payload))

    async def _safe_send(self, client: object, payload: str) -> None:
        try:
            await client.send(payload)  # type: ignore[attr-defined]
        except Exception:
            self._clients.discard(client)

    async def _handler(self, websocket: object) -> None:
        self._clients.add(websocket)
        try:
            await websocket.send(  # type: ignore[attr-defined]
                json.dumps(
                    {
                        "type": "state",
                        "state": self._session_manager.state.value,
                        "message": None,
                    }
                )
            )
            async for _ in websocket:  # type: ignore[attr-defined]
                pass
        finally:
            self._clients.discard(websocket)

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, self._host, self._port)
        log_event(logger, logging.INFO, "status_server_started", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
