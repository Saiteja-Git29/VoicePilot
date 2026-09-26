from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConnectionState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class SessionEventType(str, Enum):
    STATE_CHANGED = "state_changed"
    MODEL_AUDIO_CHUNK = "model_audio_chunk"
    TURN_COMPLETE = "turn_complete"
    INTERRUPTED = "interrupted"
    TOOL_CALL = "tool_call"  # reserved for a future computer-use phase; unused here
    ERROR = "error"


@dataclass
class SessionEvent:
    type: SessionEventType
    state: Optional[ConnectionState] = None
    data: Optional[bytes] = None
    message: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)
