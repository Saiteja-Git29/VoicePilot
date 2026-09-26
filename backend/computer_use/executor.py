from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .actions import Action

logger = logging.getLogger("voicepilot.computer_use.executor")


def _denormalize(point: Any, screen_size) -> tuple[int, int]:
    # Gemini returns [y, x] in normalized 0-1000 space.
    y, x = point
    width, height = screen_size
    return int(x / 1000 * width), int(y / 1000 * height)


def _key_map(name: str) -> str:
    return name.strip().lower().replace("cmd", "command")


class ActionExecutor:
    """Executes ALLOWED_ACTIONS locally via PyAutoGUI. Anything not already
    validated against the allowlist must never reach this class."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.3
        self._pyautogui = pyautogui

    def execute(self, action: Action) -> str:
        pyautogui = self._pyautogui
        args: Dict[str, Any] = action.args

        if action.name == "click_at":
            x, y = _denormalize(args["point"], pyautogui.size())
            pyautogui.click(x, y)
            return f"clicked at ({x}, {y})"

        if action.name == "type_text_at":
            x, y = _denormalize(args["point"], pyautogui.size())
            pyautogui.click(x, y)
            if args.get("clear_before_typing"):
                pyautogui.hotkey("command", "a")
                pyautogui.press("backspace")
            pyautogui.typewrite(args.get("text", ""), interval=0.02)
            if args.get("press_enter"):
                pyautogui.press("enter")
            return f"typed text at ({x}, {y})"

        if action.name == "key_combination":
            keys = [_key_map(k) for k in args.get("keys", "").split("+") if k.strip()]
            if keys:
                pyautogui.hotkey(*keys)
            return f"pressed keys {'+'.join(keys)}"

        if action.name == "scroll_document":
            amount = -600 if args.get("direction") == "down" else 600
            pyautogui.scroll(amount)
            return f"scrolled {args.get('direction', 'down')}"

        if action.name == "wait":
            time.sleep(min(float(args.get("seconds", 1)), 5))
            return "waited"

        if action.name == "task_complete":
            return "task marked complete"

        raise ValueError(f"executor has no handler for allowed action {action.name!r}")
