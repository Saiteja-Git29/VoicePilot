"""Regression coverage for GeminiLiveSession's receive loop.

google-genai's session.receive() yields an async generator scoped to ONE
server turn -- it ends naturally at each turn boundary, not at the end of
the connection. An earlier version of this code treated that natural end as
a fatal "stream ended" error and reconnected after every single turn, which
would have silently discarded conversation context. These tests fake only
the google.genai client boundary (never the real network) and drive
GeminiLiveSession itself, so a regression here fails loudly instead of only
showing up as a live-hardware symptom.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import List

from backend.config import VoiceConfig
from backend.voice.events import SessionEventType
from backend.voice.gemini_live_session import GeminiLiveSession


def make_config(**overrides: object) -> VoiceConfig:
    base = dict(
        api_key="test-key",
        live_model="test-model",
        api_version="v1alpha",
        voice_name="Charon",
        input_sample_rate=16000,
        output_sample_rate=24000,
        channels=1,
        chunk_size=1024,
        ws_host="127.0.0.1",
        ws_port=0,
        max_reconnect_attempts=3,
        reconnect_base_delay_seconds=0.01,
        reconnect_max_delay_seconds=0.02,
        vad_rms_threshold=500,
        vad_onset_chunks=2,
        vad_hangover_chunks=10,
        log_level="CRITICAL",
    )
    base.update(overrides)
    return VoiceConfig(**base)  # type: ignore[arg-type]


def _audio_response(data: bytes, turn_complete: bool = False) -> SimpleNamespace:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data))
    return SimpleNamespace(
        tool_call=None,
        server_content=SimpleNamespace(
            interrupted=False,
            model_turn=SimpleNamespace(parts=[part]),
            turn_complete=turn_complete,
        ),
    )


class FakeLiveSession:
    """Fakes only the google.genai Live session boundary: receive() returns a
    fresh generator per call, scoped to one turn -- exactly like the real API."""

    def __init__(self, turns: List[List[SimpleNamespace]]) -> None:
        self._turns = list(turns)
        self.sent_audio: List[bytes] = []

    async def send_realtime_input(self, audio) -> None:  # noqa: ANN001
        self.sent_audio.append(audio.data)

    async def receive(self):
        if not self._turns:
            # No more scripted turns: behave like an idle open connection
            # (hang) rather than ending, so the loop under test doesn't spin.
            await asyncio.Event().wait()
            return
        turn = self._turns.pop(0)
        for item in turn:
            yield item


class FakeLiveConnectCM:
    def __init__(self, session: FakeLiveSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeLiveSession:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeGenaiClient:
    def __init__(self, session: FakeLiveSession) -> None:
        self._session = session
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    def _connect(self, model: str, config: object) -> FakeLiveConnectCM:
        return FakeLiveConnectCM(self._session)


class GeminiLiveSessionReceiveLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_receive_loop_continues_across_turn_boundaries(self) -> None:
        fake_session = FakeLiveSession(
            [
                [_audio_response(b"turn1-audio", turn_complete=True)],
                [_audio_response(b"turn2-audio", turn_complete=True)],
            ]
        )
        session = GeminiLiveSession(FakeGenaiClient(fake_session), make_config())  # type: ignore[arg-type]

        await session.connect()
        try:
            collected = []
            events = session.events()
            for _ in range(4):  # 2 audio chunks + 2 turn_completes
                collected.append(await asyncio.wait_for(events.__anext__(), timeout=1.0))
        finally:
            await session.close()

        types_seen = [event.type for event in collected]
        self.assertEqual(
            types_seen,
            [
                SessionEventType.MODEL_AUDIO_CHUNK,
                SessionEventType.TURN_COMPLETE,
                SessionEventType.MODEL_AUDIO_CHUNK,
                SessionEventType.TURN_COMPLETE,
            ],
        )
        # The key regression check: no ERROR event between turns. A generator
        # ending at a turn boundary must not be mistaken for a dead connection.
        self.assertNotIn(SessionEventType.ERROR, types_seen)
        self.assertEqual([e.data for e in collected if e.data], [b"turn1-audio", b"turn2-audio"])

    async def test_real_disconnect_still_surfaces_as_error(self) -> None:
        class DyingLiveSession(FakeLiveSession):
            async def receive(self):
                yield _audio_response(b"only-chunk")
                raise ConnectionError("socket closed")

        fake_session = DyingLiveSession([])
        session = GeminiLiveSession(FakeGenaiClient(fake_session), make_config())  # type: ignore[arg-type]

        await session.connect()
        try:
            events = session.events()
            first = await asyncio.wait_for(events.__anext__(), timeout=1.0)
            second = await asyncio.wait_for(events.__anext__(), timeout=1.0)
        finally:
            await session.close()

        self.assertEqual(first.type, SessionEventType.MODEL_AUDIO_CHUNK)
        self.assertEqual(second.type, SessionEventType.ERROR)


if __name__ == "__main__":
    unittest.main()
