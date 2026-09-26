from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Hackathon MVP allowlist. Anything not in this set is rejected, never executed.
ALLOWED_ACTIONS = {
    "click_at",
    "type_text_at",
    "key_combination",
    "scroll_document",
    "wait",
    "task_complete",
}


@dataclass
class Action:
    name: str
    args: Dict[str, Any]


class UnsupportedActionError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unsupported computer-use action: {name!r}")
        self.name = name


def parse_action(raw: Dict[str, Any]) -> Action:
    name = raw.get("action")
    if not isinstance(name, str) or name not in ALLOWED_ACTIONS:
        raise UnsupportedActionError(str(name))
    return Action(name=name, args=raw.get("args") or {})


def validate_action(name: str) -> Optional[str]:
    """Returns None if allowed, else a human-readable rejection reason."""
    if name not in ALLOWED_ACTIONS:
        return f"'{name}' is not an allowed action for this demo"
    return None
