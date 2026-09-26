from __future__ import annotations

import asyncio
import logging
import signal
from typing import Optional

from .audio.pyaudio_io import PyAudioMicInput, PyAudioSpeakerOutput
from .config import VoiceConfig
from .logging_setup import log_event, setup_logging
from .server.ws_server import StatusServer
from .voice.gemini_live_session import GeminiLiveProvider
from .voice.session_manager import SessionManager
from .wakeword.dev_detector import KeypressWakeWordDetector
from .wakeword.gate import WakeWordGate


async def run() -> None:
    config = VoiceConfig.from_env()
    logger = setup_logging(config.log_level)

    audio_in = PyAudioMicInput(config.input_sample_rate, config.channels, config.chunk_size)
    audio_out = PyAudioSpeakerOutput(config.output_sample_rate, config.channels)
    provider = GeminiLiveProvider(config)
    # One reusable SessionManager: WakeWordGate starts/stops it per activation
    # rather than constructing a new one each time, so StatusServer's single
    # fixed reference to it stays valid across wake cycles.
    session_manager = SessionManager(provider, audio_in, audio_out, config)
    status_server = StatusServer(session_manager, config.ws_host, config.ws_port)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # Windows: no signal handlers via the event loop

    await status_server.start()

    gate: Optional[WakeWordGate] = None
    if config.wake_word_enabled:
        detector = KeypressWakeWordDetector(config.wake_words)
        gate = WakeWordGate(detector, lambda: session_manager, config.wake_word_idle_timeout_seconds)
        await gate.start()
        log_event(logger, logging.INFO, "voicepilot_started", model=config.live_model, wake_word_enabled=True)
    else:
        await session_manager.start()
        log_event(logger, logging.INFO, "voicepilot_started", model=config.live_model, wake_word_enabled=False)

    await stop_event.wait()

    log_event(logger, logging.INFO, "voicepilot_stopping")
    if gate is not None:
        await gate.stop()
    else:
        await session_manager.stop()
    await status_server.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
