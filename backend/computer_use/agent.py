from __future__ import annotations

import logging
from typing import List

from ..config import VoiceConfig
from ..logging_setup import log_event
from .actions import parse_action, validate_action
from .client import GeminiComputerUseClient
from .executor import ActionExecutor
from .screenshot import take_screenshot_png

logger = logging.getLogger("voicepilot.computer_use.agent")


class ComputerUseAgent:
    """Synchronous, blocking agent loop -- callers (SessionManager) must run
    this via a thread executor so it doesn't block the asyncio event loop
    that's also streaming mic/speaker audio."""

    def __init__(self, config: VoiceConfig) -> None:
        self._client = GeminiComputerUseClient(config)
        self._executor = ActionExecutor()
        self._max_actions = config.computer_use_max_actions

    def run(self, goal: str) -> str:
        history: List[str] = []
        for iteration in range(1, self._max_actions + 1):
            screenshot = take_screenshot_png()
            raw = self._client.next_action(goal, screenshot, history)
            action_name = str(raw.get("action"))
            rejection = validate_action(action_name)

            log_event(
                logger,
                logging.INFO,
                "COMPUTER_USE_ACTION",
                iteration=iteration,
                action=action_name,
                coordinates=(raw.get("args") or {}).get("point"),
                rejected=bool(rejection),
            )

            if rejection:
                return f"I can't do that safely ({rejection}). Stopping after {iteration} step(s)."

            action = parse_action(raw)
            if action.name == "task_complete":
                return f"Done: {goal}"

            try:
                result = self._executor.execute(action)
                log_event(logger, logging.INFO, "COMPUTER_USE_ACTION_RESULT", iteration=iteration, result=result)
                history.append(f"{action.name} -> {result}")
            except Exception as exc:
                log_event(logger, logging.ERROR, "COMPUTER_USE_ACTION_FAILED", iteration=iteration, error=str(exc))
                # Feed the failure back to the model next loop iteration instead of
                # aborting -- let it replan rather than blindly continuing.
                history.append(f"{action.name} -> FAILED: {exc}")

        return f"Stopped after {self._max_actions} actions without the model confirming completion."
