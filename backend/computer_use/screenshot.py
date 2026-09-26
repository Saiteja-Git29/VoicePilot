from __future__ import annotations

import io


def take_screenshot_png() -> bytes:
    import pyautogui

    image = pyautogui.screenshot()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
