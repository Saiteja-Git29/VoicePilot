# VoicePilot 🎙️⚡

**Voice-Driven Autonomous Computer Agent with Closed-Loop Verification**

- **Hackathon:** Google DeepMind Hyderabad Hackathon 2026
- **Track:** Problem Statement 1 — Frontier Intelligence at Flash Speed
- **Technical Reference:** [AI-Companion (Dheerajkalisetti/AI-Companion)](https://github.com/Dheerajkalisetti/AI-Companion.git)

---

## 🎯 Current Project Objective

Build an ultrafast, voice-driven autonomous computer agent operating at flash speed with end-to-end reliability, perception, and human-in-the-loop safety:

1. **Natural Language Voice Understanding:** Low-latency bidirectional audio streaming and intent recognition.
2. **Multi-Step Task Planning:** High-accuracy spatial, temporal, and desktop action decomposition.
3. **Desktop Interaction:** Native OS interaction (mouse, keyboard, shortcuts, system automation).
4. **Perception & Observation:** High-resolution screen capture, state perception, and multimodal context tracking.
5. **Closed-Loop Verification:** Visual and programmatic verification of action outcomes against user intent.
6. **Error Recovery & Re-Planning:** Autonomous detection of failures, edge cases, and unexpected state changes with adaptive correction.
7. **Barge-in / Interruptibility:** Instant voice and UI interruption allowing the user to halt or redirect the agent mid-task.
8. **Human-in-the-Loop Confirmation:** Strict policy gating requiring explicit user confirmation before executing sensitive or destructive operations.

---

## 🏗️ Architecture Overview (Planned)

```
                       +-----------------------+
                       |   User Voice / Mic    |
                       +-----------+-----------+
                                   | (Live Audio Stream)
                                   v
                       +-----------------------+
                       |   Voice Bridge        | <--- Interruption / Barge-in
                       +-----------+-----------+
                                   |
                                   v
+----------------+     +-----------------------+     +---------------------+
| Screen Monitor | --> |  Gemini Flash Agent   | <-- | Human Confirmation  |
| (Vision Loop)  |     | (Plan, Act, Verify)   |     | Guard (Safety Gate) |
+----------------+     +-----------+-----------+     +---------------------+
                                   |
                                   v
                       +-----------------------+
                       | Execution & Verify    |
                       |  - Action Dispatch    |
                       |  - Visual Verifier    |
                       |  - Self-Healing Loop  |
                       +-----------------------+
```

---

## 📁 Repository Structure

```
voicepilot/
├── README.md             # Project overview, architecture, and documentation
├── .gitignore            # Multi-stack ignore rules (Node, Python, macOS, IDE)
├── .env.example          # Environment variable template (keys never committed)
└── ...
```

---

## 🎙️ Phase 4 — Real-Time Voice Layer

The voice-first foundation (no computer-use yet): a provider-isolated realtime
client around the Gemini Live API, with genuine server-driven barge-in, a
connection state machine, reconnect/backoff, and structured logs.

```
backend/
├── audio/        # AudioInput / AudioOutput interfaces + PyAudio implementation
├── voice/        # RealtimeSession / VoiceModelProvider interfaces + Gemini Live implementation + SessionManager
├── server/       # Read-only WebSocket status broadcast for the UI
└── main.py        # Entry point
ui/                 # Minimal status page (connection/mic/speaking/interruption/error)
tests/              # Dependency-free SessionManager tests (fakes, no network/audio hardware)
```

### Setup

The venv lives one level up, at `../.venv` (workspace root), not inside this
folder — VS Code's Python extension created it there across the whole
multi-root workspace, so it's the one pinned in `.vscode/settings.json`.

```bash
python3 -m venv ../.venv          # only if it doesn't already exist
source ../.venv/bin/activate
pip install -r requirements.txt   # installs ONLY this file's deps
cp .env.example .env              # then set GEMINI_API_KEY
```

On this machine specifically, Homebrew's Python 3.14 has a broken `pyexpat`
(missing symbol vs. the OS's dyld shared cache), which breaks any `pip
install` that needs to build a wheel. Fix: `brew install expat` once, then
prefix pip commands with `DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`.
Not needed to *run* the app — only to install/rebuild dependencies.

### Run

```bash
source ../.venv/bin/activate
python3 -m backend.main          # starts the voice session + status WebSocket on :8765
open ui/index.html               # or serve ui/ statically; connects to ws://127.0.0.1:8765
```

Speak naturally once connected — no push-to-talk. Interrupting the assistant
mid-response is handled by Gemini Live's own activity detection: the mic
never stops streaming, so the server signals the interruption and VoicePilot
flushes local playback and returns to listening. A response only counts as
finished once its queued audio has actually finished playing (`drain()`),
not merely when Gemini signals `turn_complete` — those aren't the same
moment when there's still buffered audio catching up.

#### Wake word (optional, `WAKE_WORD_ENABLED=true`)

Off by default (unchanged behavior: connects to Gemini Live immediately).
When enabled, `backend/wakeword/gate.py` sits in front of the session: no
Gemini connection exists and no mic audio leaves the machine until a wake
word is detected locally. The shipped detector
(`backend/wakeword/dev_detector.py`) is a **development stand-in** — it reads
a typed line from the terminal instead of doing real keyword spotting on
audio. A real local engine (Porcupine, openWakeWord, ...) can replace it
later behind the same `WakeWordDetector` interface without touching the gate
or SessionManager.

### Test

```bash
source ../.venv/bin/activate
python3 -m unittest discover -s tests
```

---

## 🛡️ License

To be finalized (Apache 2.0 / MIT compatible clean implementation).
