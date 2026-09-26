from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types

from ..config import VoiceConfig
from .events import SessionEvent, SessionEventType
from .interfaces import RealtimeSession, VoiceModelProvider

logger = logging.getLogger("voicepilot.voice.gemini")


class GeminiLiveSession(RealtimeSession):
    """RealtimeSession backed by the Gemini Live API. Represents exactly one
    connection attempt — reconnect policy lives in SessionManager, which creates
    a fresh instance (via GeminiLiveProvider) per attempt."""

    def __init__(self, client: genai.Client, config: VoiceConfig) -> None:
        self._client = client
        self._config = config
        computer_use_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="perform_computer_task",
                    description=(
                        "Operate this computer's mouse/keyboard/browser to complete a concrete "
                        "on-screen task the user asked for, e.g. opening an app, searching the web, "
                        "clicking something, or scrolling."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "goal": types.Schema(
                                type=types.Type.STRING,
                                description="The concrete task to perform on screen.",
                            )
                        },
                        required=["goal"],
                    ),
                )
            ]
        )
        self._live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.voice_name)
                )
            ),
            system_instruction=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            "CRITICAL INSTRUCTION, follow exactly:\n"
                            "You control this Mac. You have ONE function tool available: "
                            "perform_computer_task(goal: string).\n"
                            "Any time the user's request involves the screen, an app, a browser, "
                            "clicking, typing, searching, or scrolling, your ONLY correct response is "
                            "to immediately call perform_computer_task with that request as the goal. "
                            "Do not describe what you would do. Do not say you lack access, a package, "
                            "software, or permission -- you already have everything you need via this "
                            "function. Do not ask for confirmation first. Call the function, then once "
                            "it returns a result, speak one short sentence summarizing what happened.\n"
                            "Example: user says 'open Chrome and search for Gemini' -> you immediately "
                            "call perform_computer_task(goal='open Chrome and search for Gemini'), wait "
                            "for the result, then say something like 'Done, Chrome is open with the "
                            "search results.'\n"
                            "Only refuse for: deleting files, running shell/terminal commands, "
                            "purchases, entering passwords, or sending messages/emails."
                        )
                    )
                ]
            ),
            tools=[computer_use_tool],
        )
        self._session_cm: Any = None
        self._session: Any = None
        self._event_queue: "asyncio.Queue[SessionEvent]" = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        self._closed = False

    async def connect(self) -> None:
        session_cm = self._client.aio.live.connect(
            model=self._config.live_model, config=self._live_config
        )
        session = await session_cm.__aenter__()
        self._session_cm = session_cm
        self._session = session
        self._closed = False
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._session is None:
            return
        mime_type = f"audio/pcm;rate={self._config.input_sample_rate}"
        await self._session.send_realtime_input(audio=types.Blob(data=pcm_chunk, mime_type=mime_type))

    async def _receive_loop(self) -> None:
        try:
            # session.receive() yields a generator scoped to one server turn, not
            # the whole connection — it ends naturally at each turn boundary. Loop
            # over it for the life of the session; only a raised exception (real
            # disconnect, auth error, etc.) means the connection itself is gone.
            while not self._closed:
                async for response in self._session.receive():
                    await self._handle_response(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # connection dropped, auth error, malformed stream, etc.
            if not self._closed:
                await self._event_queue.put(SessionEvent(type=SessionEventType.ERROR, message=str(exc)))

    async def _handle_response(self, response: Any) -> None:
        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            for call in getattr(tool_call, "function_calls", None) or []:
                await self._event_queue.put(
                    SessionEvent(
                        type=SessionEventType.TOOL_CALL,
                        detail={"name": call.name, "args": dict(call.args or {}), "id": call.id},
                    )
                )
        server_content = getattr(response, "server_content", None)
        if server_content is None:
            return
        if getattr(server_content, "interrupted", False):
            await self._event_queue.put(SessionEvent(type=SessionEventType.INTERRUPTED))
        model_turn = getattr(server_content, "model_turn", None)
        if model_turn is not None:
            for part in model_turn.parts or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and inline_data.data:
                    await self._event_queue.put(
                        SessionEvent(type=SessionEventType.MODEL_AUDIO_CHUNK, data=inline_data.data)
                    )
        if getattr(server_content, "turn_complete", False):
            await self._event_queue.put(SessionEvent(type=SessionEventType.TURN_COMPLETE))

    async def send_tool_response(self, call_id: str, name: str, result_text: str) -> None:
        if self._session is None:
            return
        await self._session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=call_id, name=name, response={"result": result_text})
            ]
        )

    async def close(self) -> None:
        self._closed = True
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            finally:
                self._session_cm = None
                self._session = None

    async def events(self) -> AsyncIterator[SessionEvent]:
        while True:
            event = await self._event_queue.get()
            yield event
            if event.type == SessionEventType.ERROR:
                return


class GeminiLiveProvider(VoiceModelProvider):
    """VoiceModelProvider for Gemini Live. This is the only module in VoicePilot
    that is allowed to import google.genai directly."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._client = genai.Client(
            api_key=config.api_key,
            http_options={"api_version": config.api_version},
        )

    def create_session(self) -> RealtimeSession:
        return GeminiLiveSession(self._client, self._config)
