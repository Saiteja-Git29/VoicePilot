"""Minimal MVP tests: modules import cleanly and the action allowlist rejects
anything not explicitly supported. Not a full test suite -- deliberately
small per the hackathon time budget.
"""
from __future__ import annotations

import unittest

from backend.computer_use.actions import (
    ALLOWED_ACTIONS,
    UnsupportedActionError,
    parse_action,
    validate_action,
)


class ComputerUseImportTests(unittest.TestCase):
    def test_modules_import(self) -> None:
        from backend.computer_use import agent, client, executor, screenshot  # noqa: F401


class ActionAllowlistTests(unittest.TestCase):
    def test_allowed_actions_pass_validation(self) -> None:
        for name in ALLOWED_ACTIONS:
            self.assertIsNone(validate_action(name))

    def test_unsupported_action_is_rejected(self) -> None:
        self.assertIsNotNone(validate_action("run_terminal_command"))
        self.assertIsNotNone(validate_action("delete_file"))
        self.assertIsNotNone(validate_action(""))

    def test_parse_action_raises_for_unsupported(self) -> None:
        with self.assertRaises(UnsupportedActionError):
            parse_action({"action": "run_terminal_command", "args": {}})

    def test_parse_action_succeeds_for_allowed(self) -> None:
        action = parse_action({"action": "click_at", "args": {"point": [500, 500]}})
        self.assertEqual(action.name, "click_at")
        self.assertEqual(action.args["point"], [500, 500])


if __name__ == "__main__":
    unittest.main()
