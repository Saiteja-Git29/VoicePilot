from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator, List, Optional

from .events import WakeWordEvent, WakeWordEventType
from .interfaces import WakeWordDetector


class KeypressWakeWordDetector(WakeWordDetector):
    """DEVELOPMENT/TEST DETECTOR -- NOT a real wake-word engine.

    Reads lines from stdin instead of listening to the microphone: typing one
    of the configured wake words (or anything, if none are configured) and
    pressing Enter simulates detection. This exists so the IDLE ->
    WAKE_DETECTED -> ACTIVE -> IDLE state machine can be built and tested
    correctly right now, without adding a large ML dependency or pretending
    a keyword-spotting model exists when it doesn't.

    A real local engine (e.g. Porcupine, openWakeWord) drops in later as a
    second WakeWordDetector implementation -- WakeWordGate and everything
    above it never needs to change, since they only depend on this
    interface. This detector never touches the microphone and never sends
    anything to Gemini.
    """

    def __init__(self, wake_words: List[str]) -> None:
        self._wake_words = [w.strip().lower() for w in wake_words if w.strip()]
        self._queue: "asyncio.Queue[WakeWordEvent]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        hint = ", ".join(self._wake_words) if self._wake_words else "<anything> (no WAKE_WORDS configured)"
        print(f"[wakeword:dev] type one of [{hint}] and press Enter to wake VoicePilot", file=sys.stderr)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                await asyncio.sleep(0.1)
                continue
            text = line.strip().lower()
            if not text:
                continue
            if not self._wake_words:
                matched: Optional[str] = text
            else:
                matched = next((w for w in self._wake_words if w in text), None)
            if matched:
                await self._queue.put(WakeWordEvent(type=WakeWordEventType.DETECTED, word=matched))

    async def events(self) -> AsyncIterator[WakeWordEvent]:
        while True:
            yield await self._queue.get()
