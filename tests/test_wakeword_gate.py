"""Tests for WakeWordGate's IDLE <-> ACTIVE lifecycle. Drives a real
SessionManager (via the existing fakes) so these prove genuine integration,
not just the gate's internal bookkeeping in isolation.
"""
from __future__ import annotations

import asyncio
import unittest
from typing import AsyncIterator, List

from backend.voice.session_manager import SessionManager
from backend.wakeword.events import WakeWordEvent, WakeWordEventType
from backend.wakeword.gate import WakeGateState, WakeWordGate
from backend.wakeword.interfaces import WakeWordDetector

from test_session_manager import FakeAudioInput, FakeAudioOutput, FakeProvider, FakeSession, make_config


class FakeWakeWordDetector(WakeWordDetector):
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self._queue: "asyncio.Queue[WakeWordEvent]" = asyncio.Queue()

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    async def events(self) -> AsyncIterator[WakeWordEvent]:
        while True:
            yield await self._queue.get()

    async def trigger(self, word: str = "hey jarvis") -> None:
        await self._queue.put(WakeWordEvent(type=WakeWordEventType.DETECTED, word=word))


class WakeWordGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_stays_idle_and_gemini_not_connected_until_wake_word(self) -> None:
        detector = FakeWakeWordDetector()
        session = FakeSession([])
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), FakeAudioOutput(), make_config())
        gate = WakeWordGate(detector, lambda: manager, idle_timeout_seconds=1.0)

        await gate.start()
        await asyncio.sleep(0.05)

        self.assertEqual(gate.state, WakeGateState.IDLE)
        self.assertEqual(detector.start_calls, 1)
        self.assertFalse(session.connect_called)  # no Gemini connection while idle

        await detector.trigger("hey jarvis")
        await asyncio.sleep(0.05)

        self.assertEqual(gate.state, WakeGateState.ACTIVE)
        self.assertTrue(session.connect_called)
        self.assertEqual(detector.stop_calls, 1)  # detector stopped once session activated

        await gate.stop()

    async def test_conversation_idle_timeout_returns_to_wake_word_wait(self) -> None:
        detector = FakeWakeWordDetector()
        session = FakeSession([])
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), FakeAudioOutput(), make_config())
        gate = WakeWordGate(detector, lambda: manager, idle_timeout_seconds=0.05)

        await gate.start()
        await detector.trigger()
        await asyncio.sleep(0.03)
        self.assertEqual(gate.state, WakeGateState.ACTIVE)

        await asyncio.sleep(0.2)  # exceed idle_timeout_seconds with no further activity

        self.assertEqual(gate.state, WakeGateState.IDLE)
        self.assertGreaterEqual(detector.start_calls, 2)  # listening for the wake word again

        await gate.stop()

    async def test_can_wake_multiple_times_with_the_same_reusable_session(self) -> None:
        detector = FakeWakeWordDetector()
        session1 = FakeSession([])
        session2 = FakeSession([])
        manager = SessionManager(FakeProvider([session1, session2]), FakeAudioInput(), FakeAudioOutput(), make_config())
        gate = WakeWordGate(detector, lambda: manager, idle_timeout_seconds=0.05)

        await gate.start()
        await detector.trigger("hey jarvis")
        await asyncio.sleep(0.03)
        self.assertTrue(session1.connect_called)

        await asyncio.sleep(0.2)
        self.assertEqual(gate.state, WakeGateState.IDLE)

        await detector.trigger("computer")
        await asyncio.sleep(0.03)

        self.assertTrue(session2.connect_called)
        self.assertEqual(gate.state, WakeGateState.ACTIVE)

        await gate.stop()


if __name__ == "__main__":
    unittest.main()
