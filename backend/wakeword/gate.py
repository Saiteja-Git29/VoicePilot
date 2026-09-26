from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import Enum
from typing import Callable, Optional

from ..logging_setup import log_event
from ..voice.events import ConnectionState
from ..voice.session_manager import SessionManager
from .events import WakeWordEventType
from .interfaces import WakeWordDetector

logger = logging.getLogger("voicepilot.wakeword.gate")


class WakeGateState(str, Enum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    ACTIVE = "active"


class WakeWordGate:
    """Owns the IDLE <-> ACTIVE lifecycle around a SessionManager.

    While IDLE, only the local WakeWordDetector is running -- no Gemini
    connection exists and no microphone audio leaves the machine. On a
    detected wake word, a fresh SessionManager (built by `session_factory`)
    is started for the conversation; once `idle_timeout_seconds` passes with
    no state change on either side (no user speech, no assistant speech),
    the session is stopped and the gate returns to IDLE, ready for the wake
    word again.
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        session_factory: Callable[[], SessionManager],
        idle_timeout_seconds: float,
        on_gate_state_change: Optional[Callable[[WakeGateState], None]] = None,
    ) -> None:
        self._detector = detector
        self._session_factory = session_factory
        self._idle_timeout_seconds = idle_timeout_seconds
        self._on_gate_state_change = on_gate_state_change
        self._state = WakeGateState.IDLE
        self._run_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> WakeGateState:
        return self._state

    def _set_state(self, state: WakeGateState) -> None:
        if state == self._state:
            return
        self._state = state
        log_event(logger, logging.INFO, "wakeword_gate_state_changed", state=state.value)
        if self._on_gate_state_change is not None:
            self._on_gate_state_change(state)

    async def start(self) -> None:
        self._run_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        await self._detector.stop()

    async def _run_loop(self) -> None:
        while True:
            await self._wait_for_wake_word()
            self._set_state(WakeGateState.WAKE_DETECTED)
            session = self._session_factory()
            await session.start()
            self._set_state(WakeGateState.ACTIVE)
            try:
                await self._run_until_idle(session)
            finally:
                await session.stop()
            self._set_state(WakeGateState.IDLE)

    async def _wait_for_wake_word(self) -> None:
        self._set_state(WakeGateState.IDLE)
        await self._detector.start()
        try:
            async for event in self._detector.events():
                if event.type == WakeWordEventType.DETECTED:
                    log_event(logger, logging.INFO, "wake_word_detected", word=event.word)
                    return
        finally:
            await self._detector.stop()

    async def _run_until_idle(self, session: SessionManager) -> None:
        """Idle means no state change at all for a full window -- not just
        ConnectionState.IDLE, which SessionManager never actually enters
        (its own IDLE is a startup-only value). Any transition (LISTENING,
        SPEAKING, an interruption, ...) resets the window."""
        activity = asyncio.Event()

        def _on_state_change(state: ConnectionState, message: Optional[str]) -> None:
            activity.set()

        session.on_state_change(_on_state_change)
        while True:
            activity.clear()
            try:
                await asyncio.wait_for(activity.wait(), timeout=self._idle_timeout_seconds)
            except asyncio.TimeoutError:
                log_event(logger, logging.INFO, "conversation_idle_timeout")
                return
