from __future__ import annotations

import abc
from typing import AsyncIterator


class AudioInput(abc.ABC):
    """Continuous microphone source. Streams full-duplex regardless of playback state —
    barge-in detection depends on the model always hearing the user, not on a local mute gate."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def frames(self) -> AsyncIterator[bytes]: ...


class AudioOutput(abc.ABC):
    """Speaker sink. `flush` must drop any buffered/in-flight audio immediately so a
    server-signaled interruption is audible within one playback chunk, not after the
    current turn finishes queuing."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def enqueue(self, pcm_chunk: bytes) -> None: ...

    @abc.abstractmethod
    def flush(self) -> None: ...

    @abc.abstractmethod
    async def drain(self) -> None:
        """Wait until every chunk enqueued so far has actually finished playing.
        A model's turn_complete means generation finished, not that local
        playback has caught up — callers must await this before treating the
        turn as over, otherwise trailing audio can be reported as done before
        it's actually heard."""
        ...
