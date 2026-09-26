from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WakeWordEventType(str, Enum):
    DETECTED = "detected"
    ERROR = "error"


@dataclass
class WakeWordEvent:
    type: WakeWordEventType
    word: Optional[str] = None
    message: Optional[str] = None
