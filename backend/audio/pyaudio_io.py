from __future__ import annotations

import asyncio
import contextlib
import functools
from typing import AsyncIterator, Optional

import pyaudio

from .interfaces import AudioInput, AudioOutput


class PyAudioMicInput(AudioInput):
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_size: int,
        device_index: Optional[int] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=50)
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=self._device_index,
        )
        self._running = True
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()
        read = functools.partial(self._stream.read, self._chunk_size, exception_on_overflow=False)
        while self._running:
            data = await loop.run_in_executor(None, read)
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    async def stop(self) -> None:
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._queue.get()


class PyAudioSpeakerOutput(AudioOutput):
    _WRITE_FRAME_SIZE = 480  # ~20ms at 24kHz mono: keeps flush() latency low

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        device_index: Optional[int] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device_index = device_index
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._flush_requested = False

    async def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._device_index,
        )
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        bytes_per_frame = 2 * self._channels
        frame_bytes = self._WRITE_FRAME_SIZE * bytes_per_frame
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                self._queue.task_done()
                return
            self._flush_requested = False
            for offset in range(0, len(chunk), frame_bytes):
                if self._flush_requested:
                    break
                piece = chunk[offset : offset + frame_bytes]
                await loop.run_in_executor(None, self._stream.write, piece)
            self._queue.task_done()

    async def enqueue(self, pcm_chunk: bytes) -> None:
        await self._queue.put(pcm_chunk)

    def flush(self) -> None:
        self._flush_requested = True
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def drain(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
