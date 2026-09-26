"""Dependency-free tests for SessionManager: exercise it against fakes of
AudioInput/AudioOutput/RealtimeSession/VoiceModelProvider so the state
machine, interruption handling, and reconnect policy are verified without
needing pyaudio, google-genai, or real network/audio hardware.
"""
from __future__ import annotations

import asyncio
import unittest
from typing import List, Optional

from backend.audio.interfaces import AudioInput, AudioOutput
from backend.config import VoiceConfig
from backend.voice.events import ConnectionState, SessionEvent, SessionEventType
from backend.voice.interfaces import RealtimeSession, VoiceModelProvider
from backend.voice.session_manager import SessionManager


class FakeAudioInput(AudioInput):
    def __init__(self, chunks: Optional[List[bytes]] = None) -> None:
        self._chunks = list(chunks or [])
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def frames(self):
        for chunk in self._chunks:
            yield chunk


class FakeAudioOutput(AudioOutput):
    """drain() is gated by an asyncio.Event so tests can prove SessionManager
    actually waits for playback to catch up, instead of just calling drain()
    and ignoring the result. The gate starts open (instant playback) unless a
    test calls block_drain()."""

    def __init__(self) -> None:
        self.enqueued: List[bytes] = []
        self.flush_called = False
        self.started = False
        self.stopped = False
        self.drain_calls = 0
        self._drain_gate = asyncio.Event()
        self._drain_gate.set()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def enqueue(self, pcm_chunk: bytes) -> None:
        self.enqueued.append(pcm_chunk)

    def flush(self) -> None:
        self.flush_called = True

    async def drain(self) -> None:
        self.drain_calls += 1
        await self._drain_gate.wait()

    def block_drain(self) -> None:
        self._drain_gate.clear()

    def release_drain(self) -> None:
        self._drain_gate.set()


class FakeSession(RealtimeSession):
    def __init__(self, events_to_emit: List[SessionEvent], connect_error: Optional[Exception] = None) -> None:
        self._events_to_emit = list(events_to_emit)
        self._connect_error = connect_error
        self.connect_called = False
        self.closed = False
        self.sent_audio: List[bytes] = []

    async def connect(self) -> None:
        self.connect_called = True
        if self._connect_error is not None:
            raise self._connect_error

    async def send_audio(self, pcm_chunk: bytes) -> None:
        self.sent_audio.append(pcm_chunk)

    async def close(self) -> None:
        self.closed = True

    async def events(self):
        for event in self._events_to_emit:
            yield event
        # Real connections stay open with no more events until closed;
        # mirror that instead of ending the generator (which would trigger reconnect).
        hang = asyncio.Event()
        await hang.wait()


class FakeProvider(VoiceModelProvider):
    def __init__(self, sessions: List[FakeSession]) -> None:
        self._sessions = list(sessions)

    def create_session(self) -> RealtimeSession:
        return self._sessions.pop(0)


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


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_connect_transitions_to_listening(self) -> None:
        session = FakeSession([])
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), FakeAudioOutput(), make_config())
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertIn(ConnectionState.CONNECTING, states)
        self.assertIn(ConnectionState.LISTENING, states)
        await manager.stop()

    async def test_model_audio_chunk_transitions_to_speaking_and_is_played(self) -> None:
        session = FakeSession([SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"abc")])
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertIn(ConnectionState.SPEAKING, states)
        self.assertEqual(audio_out.enqueued, [b"abc"])
        await manager.stop()

    async def test_interruption_flushes_output_and_returns_to_listening(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"abc"),
                SessionEvent(type=SessionEventType.INTERRUPTED),
            ]
        )
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())
        transitions: List[tuple] = []
        manager.on_state_change(lambda s, m: transitions.append((s, m)))

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertTrue(audio_out.flush_called)
        self.assertEqual(transitions[-1], (ConnectionState.LISTENING, "interrupted"))
        await manager.stop()

    async def test_turn_complete_returns_to_listening(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"abc"),
                SessionEvent(type=SessionEventType.TURN_COMPLETE),
            ]
        )
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), FakeAudioOutput(), make_config())
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertEqual(states[-1], ConnectionState.LISTENING)
        await manager.stop()

    async def test_reconnects_after_connect_failure_then_succeeds(self) -> None:
        failing = FakeSession([], connect_error=RuntimeError("boom"))
        succeeding = FakeSession([])
        config = make_config(max_reconnect_attempts=3, reconnect_base_delay_seconds=0.01, reconnect_max_delay_seconds=0.01)
        manager = SessionManager(FakeProvider([failing, succeeding]), FakeAudioInput(), FakeAudioOutput(), config)
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.1)

        self.assertTrue(failing.connect_called)
        self.assertTrue(succeeding.connect_called)
        self.assertIn(ConnectionState.RECONNECTING, states)
        self.assertEqual(states[-1], ConnectionState.LISTENING)
        await manager.stop()

    async def test_gives_up_after_max_reconnect_attempts(self) -> None:
        always_failing = [FakeSession([], connect_error=RuntimeError("boom")) for _ in range(5)]
        config = make_config(max_reconnect_attempts=2, reconnect_base_delay_seconds=0.01, reconnect_max_delay_seconds=0.01)
        manager = SessionManager(FakeProvider(always_failing), FakeAudioInput(), FakeAudioOutput(), config)
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.2)

        self.assertEqual(states[-1], ConnectionState.ERROR)
        await manager.stop()

    async def test_mic_audio_is_forwarded_to_session(self) -> None:
        session = FakeSession([])
        audio_in = FakeAudioInput([b"chunk1", b"chunk2"])
        manager = SessionManager(FakeProvider([session]), audio_in, FakeAudioOutput(), make_config())

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertEqual(session.sent_audio, [b"chunk1", b"chunk2"])
        await manager.stop()

    # --- Phase 4.1: audio playback lifecycle -------------------------------

    async def test_multiple_model_audio_chunks_all_forwarded(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"a"),
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"b"),
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"c"),
            ]
        )
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertEqual(audio_out.enqueued, [b"a", b"b", b"c"])
        await manager.stop()

    async def test_turn_complete_does_not_discard_pending_audio(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"a"),
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"b"),
                SessionEvent(type=SessionEventType.TURN_COMPLETE),
            ]
        )
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())

        await manager.start()
        await asyncio.sleep(0.05)

        # turn_complete must never discard already-enqueued audio via flush().
        self.assertEqual(audio_out.enqueued, [b"a", b"b"])
        self.assertFalse(audio_out.flush_called)
        await manager.stop()

    async def test_playback_completion_waits_for_drain_before_listening(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"a"),
                SessionEvent(type=SessionEventType.TURN_COMPLETE),
            ]
        )
        audio_out = FakeAudioOutput()
        audio_out.block_drain()  # simulate playback still catching up
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.05)

        # turn_complete arrived, but drain() hasn't resolved yet -> still SPEAKING.
        self.assertEqual(audio_out.drain_calls, 1)
        self.assertEqual(states[-1], ConnectionState.SPEAKING)

        audio_out.release_drain()
        await asyncio.sleep(0.05)

        # only now, after drain() resolves, does the turn actually end.
        self.assertEqual(states[-1], ConnectionState.LISTENING)
        await manager.stop()

    async def test_interruption_still_flushes_pending_audio(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"a"),
                SessionEvent(type=SessionEventType.INTERRUPTED),
            ]
        )
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertTrue(audio_out.flush_called)
        # interruption must NOT wait on drain() -- that would defeat barge-in.
        self.assertEqual(audio_out.drain_calls, 0)
        await manager.stop()

    async def test_session_usable_and_subsequent_response_played_after_completion(self) -> None:
        session = FakeSession(
            [
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"turn1-a"),
                SessionEvent(type=SessionEventType.TURN_COMPLETE),
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"turn2-a"),
                SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=b"turn2-b"),
                SessionEvent(type=SessionEventType.TURN_COMPLETE),
            ]
        )
        audio_out = FakeAudioOutput()
        manager = SessionManager(FakeProvider([session]), FakeAudioInput(), audio_out, make_config())
        states: List[ConnectionState] = []
        manager.on_state_change(lambda s, m: states.append(s))

        await manager.start()
        await asyncio.sleep(0.05)

        self.assertEqual(audio_out.enqueued, [b"turn1-a", b"turn2-a", b"turn2-b"])
        self.assertEqual(audio_out.drain_calls, 2)
        self.assertEqual(states[-1], ConnectionState.LISTENING)
        # both SPEAKING and LISTENING appear more than once -> session cycled
        # through a second full turn without being torn down or reconnected.
        self.assertEqual(states.count(ConnectionState.SPEAKING), 2)
        await manager.stop()


if __name__ == "__main__":
    unittest.main()
