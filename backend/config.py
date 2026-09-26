from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class VoiceConfig:
    api_key: str
    live_model: str
    api_version: str
    voice_name: str
    input_sample_rate: int
    output_sample_rate: int
    channels: int
    chunk_size: int
    ws_host: str
    ws_port: int
    max_reconnect_attempts: int
    reconnect_base_delay_seconds: float
    reconnect_max_delay_seconds: float
    vad_rms_threshold: float
    vad_onset_chunks: int
    vad_hangover_chunks: int
    wake_word_enabled: bool = False
    wake_words: List[str] = field(default_factory=list)
    wake_word_idle_timeout_seconds: float = 30.0
    computer_use_model: str = "gemini-3.8-flash"
    computer_use_max_actions: int = 10
    log_level: str = "INFO"

    @staticmethod
    def from_env() -> "VoiceConfig":
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return VoiceConfig(
            api_key=api_key,
            live_model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live"),
            api_version=os.getenv("GEMINI_LIVE_API_VERSION", "v1alpha"),
            voice_name=os.getenv("GEMINI_VOICE_NAME", "Charon"),
            input_sample_rate=_int("VOICE_INPUT_SAMPLE_RATE", 16000),
            output_sample_rate=_int("VOICE_OUTPUT_SAMPLE_RATE", 24000),
            channels=_int("VOICE_CHANNELS", 1),
            chunk_size=_int("VOICE_CHUNK_SIZE", 1024),
            ws_host=os.getenv("VOICE_WS_HOST", "127.0.0.1"),
            ws_port=_int("VOICE_WS_PORT", 8765),
            max_reconnect_attempts=_int("VOICE_MAX_RECONNECT_ATTEMPTS", 5),
            reconnect_base_delay_seconds=float(os.getenv("VOICE_RECONNECT_BASE_DELAY", "1.0")),
            reconnect_max_delay_seconds=float(os.getenv("VOICE_RECONNECT_MAX_DELAY", "30.0")),
            vad_rms_threshold=float(os.getenv("VOICE_VAD_RMS_THRESHOLD", "500")),
            vad_onset_chunks=_int("VOICE_VAD_ONSET_CHUNKS", 2),
            vad_hangover_chunks=_int("VOICE_VAD_HANGOVER_CHUNKS", 10),
            wake_word_enabled=_bool("WAKE_WORD_ENABLED", False),
            wake_words=[w for w in os.getenv("WAKE_WORDS", "hey jarvis,computer").split(",") if w.strip()],
            wake_word_idle_timeout_seconds=float(os.getenv("WAKE_WORD_IDLE_TIMEOUT_SECONDS", "30")),
            computer_use_model=os.getenv("GEMINI_COMPUTER_USE_MODEL", "gemini-3.8-flash"),
            computer_use_max_actions=_int("COMPUTER_USE_MAX_ACTIONS", 10),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
