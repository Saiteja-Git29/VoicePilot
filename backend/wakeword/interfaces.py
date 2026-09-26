from __future__ import annotations

import abc
from typing import AsyncIterator

from .events import WakeWordEvent


class WakeWordDetector(abc.ABC):
    """Provider-independent local wake-word listener. Implementations own
    whatever they need to listen for a wake word (their own mic stream, a
    keyword-spotting model, a manual trigger, ...) WITHOUT the main
    conversational audio session existing yet. Nothing here may depend on
    Gemini or send audio anywhere -- idle listening must stay fully local.
    Swapping in a real engine (Porcupine, openWakeWord, ...) means
    implementing this interface only; WakeWordGate never changes."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[WakeWordEvent]: ...
