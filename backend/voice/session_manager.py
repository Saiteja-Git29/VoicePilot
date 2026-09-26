from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from typing import Callable, List, Optional

from ..audio.interfaces import AudioInput, AudioOutput
from ..config import VoiceConfig
from ..logging_setup import log_event
from .events import ConnectionState, SessionEvent, SessionEventType
from .interfaces import RealtimeSession, VoiceModelProvider

logger = logging.getLogger("voicepilot.voice.session_manager")

StateListener = Callable[[ConnectionState, Optional[str]], None]


class _GiveUp(Exception):
    """Raised internally when reconnect attempts are exhausted."""


def _rms(pcm_bytes: bytes) -> float:
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
    return (sum(s * s for s in samples) / count) ** 0.5


class SessionManager:
    """Owns the full voice pipeline lifecycle: audio I/O, the realtime session,
    reconnect policy, and the connection state machine. This is the only class
    that should know both "audio" and "voice provider" exist — everything else
    (the WS/UI layer) only sees state + events."""

    def __init__(
        self,
        provider: VoiceModelProvider,
        audio_in: AudioInput,
        audio_out: AudioOutput,
        config: VoiceConfig,
    ) -> None:
        self._provider = provider
        self._audio_in = audio_in
        self._audio_out = audio_out
        self._config = config

        self._state = ConnectionState.IDLE
        self._listeners: List[StateListener] = []

        self._session: Optional[RealtimeSession] = None
        self._mic_task: Optional[asyncio.Task] = None
        self._run_task: Optional[asyncio.Task] = None
        self._stopping = False

        self._speaking_turn_open = False
        self._voice_active = False
        self._speech_run = 0
        self._silence_run = 0
        self._computer_use_agent = None

    @property
    def state(self) -> ConnectionState:
        return self._state

    def on_state_change(self, listener: StateListener) -> None:
        self._listeners.append(listener)

    def _set_state(self, state: ConnectionState, message: Optional[str] = None) -> None:
        if state == self._state:
            return
        self._state = state
        log_event(logger, logging.INFO, "state_changed", state=state.value, message=message)
        for listener in self._listeners:
            listener(state, message)

    async def start(self) -> None:
        self._stopping = False
        try:
            await self._audio_in.start()
            log_event(logger, logging.INFO, "microphone_started")
        except Exception as exc:
            log_event(logger, logging.ERROR, "microphone_unavailable", error=str(exc))
            self._set_state(ConnectionState.ERROR, f"microphone unavailable: {exc}")
            raise
        try:
            await self._audio_out.start()
            log_event(logger, logging.INFO, "speaker_started")
        except Exception as exc:
            log_event(logger, logging.ERROR, "speaker_unavailable", error=str(exc))
            self._set_state(ConnectionState.ERROR, f"speaker unavailable: {exc}")
            raise
        self._run_task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stopping = True
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        await self._teardown_session()
        await self._audio_in.stop()
        log_event(logger, logging.INFO, "microphone_stopped")
        await self._audio_out.stop()
        log_event(logger, logging.INFO, "speaker_stopped")
        self._set_state(ConnectionState.CLOSED)

    async def _run_forever(self) -> None:
        while not self._stopping:
            try:
                await self._connect_with_retry()
            except _GiveUp:
                self._set_state(ConnectionState.ERROR, "max reconnect attempts exceeded")
                return
            if self._stopping:
                return
            await self._serve_session()
            if self._stopping:
                return
            self._set_state(ConnectionState.RECONNECTING)

    async def _connect_with_retry(self) -> None:
        attempt = 0
        while True:
            self._set_state(ConnectionState.CONNECTING if attempt == 0 else ConnectionState.RECONNECTING)
            session = self._provider.create_session()
            try:
                await session.connect()
            except Exception as exc:
                attempt += 1
                log_event(
                    logger, logging.WARNING, "session_connect_failed", attempt=attempt, error=str(exc)
                )
                if attempt > self._config.max_reconnect_attempts:
                    raise _GiveUp() from exc
                delay = min(
                    self._config.reconnect_base_delay_seconds * (2 ** (attempt - 1)),
                    self._config.reconnect_max_delay_seconds,
                )
                await asyncio.sleep(delay)
                continue
            self._session = session
            log_event(logger, logging.INFO, "session_connected", attempt=attempt)
            self._set_state(ConnectionState.LISTENING)
            return

    async def _serve_session(self) -> None:
        assert self._session is not None
        self._mic_task = asyncio.create_task(self._pump_mic())
        try:
            async for event in self._session.events():
                await self._handle_event(event)
        finally:
            if self._mic_task is not None:
                self._mic_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._mic_task
                self._mic_task = None
            await self._teardown_session()
            log_event(logger, logging.INFO, "session_disconnected")

    async def _pump_mic(self) -> None:
        async for chunk in self._audio_in.frames():
            self._update_local_vad(chunk)
            if self._session is None:
                return
            outgoing = chunk
            if self._state == ConnectionState.SPEAKING:
                # Without headphones/AEC, the mic hears our own speaker output and
                # Gemini's server-side VAD (correctly) treats that as the user
                # interrupting -- cutting every response off within ~300ms. Muting
                # the outgoing stream while WE'RE the one talking fixes that; the
                # trade-off is genuine barge-in only re-engages once this turn ends.
                outgoing = b"\x00" * len(chunk)
            try:
                await self._session.send_audio(outgoing)
            except Exception as exc:
                log_event(logger, logging.ERROR, "send_audio_failed", error=str(exc))
                return

    def _update_local_vad(self, chunk: bytes) -> None:
        """Cheap RMS-based voice-activity estimate, used only for observability
        logs (user_turn_started/completed). It never gates what gets streamed —
        the mic keeps sending continuously so the model's own server-side
        activity detection can genuinely handle barge-in."""
        level = _rms(chunk)
        if level >= self._config.vad_rms_threshold:
            self._speech_run += 1
            self._silence_run = 0
            if not self._voice_active and self._speech_run >= self._config.vad_onset_chunks:
                self._voice_active = True
                log_event(logger, logging.INFO, "user_turn_started")
        else:
            self._silence_run += 1
            self._speech_run = 0
            if self._voice_active and self._silence_run >= self._config.vad_hangover_chunks:
                self._voice_active = False
                log_event(logger, logging.INFO, "user_turn_completed")

    async def _handle_event(self, event: SessionEvent) -> None:
        if event.type == SessionEventType.MODEL_AUDIO_CHUNK and event.data is not None:
            if not self._speaking_turn_open:
                self._speaking_turn_open = True
                log_event(logger, logging.INFO, "assistant_turn_started")
            self._set_state(ConnectionState.SPEAKING)
            await self._audio_out.enqueue(event.data)
        elif event.type == SessionEventType.INTERRUPTED:
            log_event(logger, logging.INFO, "interruption")
            self._audio_out.flush()
            self._speaking_turn_open = False
            self._set_state(ConnectionState.LISTENING, "interrupted")
        elif event.type == SessionEventType.TURN_COMPLETE:
            # turn_complete means the MODEL finished generating, not that local
            # playback has caught up — wait for the speaker queue to actually
            # drain before declaring the turn (and the state) over.
            await self._audio_out.drain()
            log_event(logger, logging.INFO, "playback_complete")
            log_event(logger, logging.INFO, "assistant_turn_completed")
            self._speaking_turn_open = False
            self._set_state(ConnectionState.LISTENING)
        elif event.type == SessionEventType.TOOL_CALL:
            detail = event.detail or {}
            if detail.get("name") == "perform_computer_task":
                asyncio.create_task(self._run_computer_use(detail))
            else:
                log_event(logger, logging.WARNING, "tool_call_received_unexpected", detail=detail)
        elif event.type == SessionEventType.ERROR:
            log_event(logger, logging.ERROR, "session_error", message=event.message)

    async def _run_computer_use(self, detail: dict) -> None:
        """Runs the (blocking, synchronous) ComputerUseAgent loop off the event
        loop thread so mic/speaker streaming keeps working while it executes,
        then reports the result back through the same Live tool-call turn."""
        goal = (detail.get("args") or {}).get("goal", "")
        call_id = detail.get("id")
        log_event(logger, logging.INFO, "computer_use_task_started", goal=goal)
        try:
            agent = self._get_computer_use_agent()
            loop = asyncio.get_running_loop()
            result_text = await loop.run_in_executor(None, agent.run, goal)
        except Exception as exc:
            result_text = f"Computer-use task failed: {exc}"
            log_event(logger, logging.ERROR, "computer_use_task_failed", error=str(exc))
        log_event(logger, logging.INFO, "computer_use_task_finished", result=result_text)
        if self._session is not None and call_id:
            await self._session.send_tool_response(call_id, "perform_computer_task", result_text)

    def _get_computer_use_agent(self):
        if self._computer_use_agent is None:
            from ..computer_use.agent import ComputerUseAgent

            self._computer_use_agent = ComputerUseAgent(self._config)
        return self._computer_use_agent

    async def _teardown_session(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
