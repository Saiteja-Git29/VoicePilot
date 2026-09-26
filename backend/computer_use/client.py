from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from google import genai
from google.genai import types

from ..config import VoiceConfig

logger = logging.getLogger("voicepilot.computer_use.client")

SYSTEM_PROMPT = """You are a macOS computer-use agent controlling this machine via mouse/keyboard.
Given a goal, the current screenshot, and the history of actions already taken, decide the SINGLE
next action to make progress. Respond with ONLY a JSON object, no markdown fences, no prose:

{"action": "<one of: click_at, type_text_at, key_combination, scroll_document, wait, task_complete>",
 "args": {...},
 "reasoning": "<short reason>"}

Argument shapes:
- click_at: {"point": [y, x]}            (y,x normalized 0-1000 over the screenshot)
- type_text_at: {"point": [y, x], "text": "...", "press_enter": true|false, "clear_before_typing": true|false}
- key_combination: {"keys": "Command+Space"}
- scroll_document: {"direction": "down"|"up"}
- wait: {"seconds": 1}
- task_complete: {} -- use this once the goal has been visibly achieved on screen.

Only ever return one of those six action names. If the goal is already achieved, return task_complete.
"""


class GeminiComputerUseClient:
    """Only module besides gemini_live_session.py that talks to google.genai.
    One call per step: given the goal + latest screenshot + action history, ask
    for exactly one next action as strict JSON."""

    def __init__(self, config: VoiceConfig) -> None:
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.computer_use_model

    def next_action(self, goal: str, screenshot_png: bytes, history: List[str]) -> Dict[str, Any]:
        history_text = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history)) or "(none yet)"
        prompt = f"Goal: {goal}\n\nActions taken so far:\n{history_text}\n\nCurrent screenshot attached."
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + prompt),
                        types.Part.from_bytes(data=screenshot_png, mime_type="image/png"),
                    ],
                )
            ],
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("computer_use_unparsable_response", extra={"fields": {"raw": text[:300]}})
            return {"action": "task_complete", "args": {}, "reasoning": "model response was unparsable"}
