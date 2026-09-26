const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_DELAY_MS = 3000;

const stateBadge = document.getElementById("state-badge");
const connectionStatus = document.getElementById("connection-status");
const micStatus = document.getElementById("mic-status");
const speakingStatus = document.getElementById("speaking-status");
const interruptStatus = document.getElementById("interrupt-status");
const errorMessage = document.getElementById("error-message");

function applyState(state, message) {
  stateBadge.textContent = state;
  stateBadge.dataset.state = state;

  connectionStatus.textContent =
    state === "idle" || state === "closed" ? "disconnected" : state;
  micStatus.textContent = state === "closed" ? "off" : "listening";
  speakingStatus.textContent = state === "speaking" ? "speaking" : "silent";
  interruptStatus.textContent = message === "interrupted" ? "interrupted" : "none";
  errorMessage.textContent = state === "error" ? message || "unknown error" : "";
}

function connect() {
  const socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    connectionStatus.textContent = "connected (waiting for status)";
  });

  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "state") {
        applyState(payload.state, payload.message);
      }
    } catch (err) {
      console.error("Malformed status message", err);
    }
  });

  socket.addEventListener("close", () => {
    applyState("closed", "backend disconnected");
    setTimeout(connect, RECONNECT_DELAY_MS);
  });

  socket.addEventListener("error", () => {
    socket.close();
  });
}

connect();
