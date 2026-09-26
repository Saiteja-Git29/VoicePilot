from __future__ import annotations

import abc
from typing import AsyncIterator

from .events import SessionEvent


class RealtimeSession(abc.ABC):
    """One live conversation with a voice model. Implementations own their own
    reconnect policy is NOT assumed here — that lives one layer up, in whatever
    drives the session, so a session object always represents a single connection
    attempt's lifecycle."""

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, pcm_chunk: bytes) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[SessionEvent]: ...

    async def send_tool_response(self, call_id: str, name: str, result_text: str) -> None:
        """Default no-op so existing fakes/providers without tool support keep
        working unchanged. Providers that declare tools (e.g. Gemini Live's
        perform_computer_task) override this."""
        return


class VoiceModelProvider(abc.ABC):
    """Factory boundary that keeps the rest of VoicePilot decoupled from any one
    vendor's realtime SDK. Swapping providers means implementing this and
    RealtimeSession only — nothing else in the app should import a provider SDK."""

    @abc.abstractmethod
    def create_session(self) -> RealtimeSession: ...
