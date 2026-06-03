"""Unix domain socket server for external LLM communication."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from typing import TYPE_CHECKING, Callable

from snaptui.model import Msg, Sub
from data_viewer.messages import SocketMsg

if TYPE_CHECKING:
    from data_viewer.app import AppModel

SOCKET_PATH = "/tmp/parrhesia-viewer.sock"


def _cleanup_socket() -> None:
    """Remove socket file if it exists."""
    try:
        os.unlink(SOCKET_PATH)
    except OSError:
        pass


def _handle_client(conn: socket.socket, send: Callable[[Msg], None], app_model: 'AppModel') -> None:
    """Handle a single client connection."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            # Process complete lines
            while b"\n" in data:
                line, data = data.split(b"\n", 1)
                _process_line(line, conn, send, app_model)

        # Handle case where data doesn't end with newline
        data = data.strip()
        if data:
            _process_line(data, conn, send, app_model)
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        conn.close()


def _process_line(line: bytes, conn: socket.socket, send: Callable[[Msg], None], app_model: 'AppModel') -> None:
    """Process a single line of input from a client."""
    line = line.strip()
    if not line:
        return
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        conn.sendall(json.dumps({"error": "invalid JSON"}).encode() + b"\n")
        return

    action = msg.get("action", "")

    if action == "status":
        status = app_model.get_status()
        conn.sendall(json.dumps(status).encode() + b"\n")
    elif action == "navigate":
        send(SocketMsg(
            action="navigate",
            data={k: v for k, v in msg.items() if k != "action"},
        ))
        conn.sendall(json.dumps({"ok": True}).encode() + b"\n")
    else:
        conn.sendall(json.dumps({"error": f"unknown action: {action}"}).encode() + b"\n")


def make_socket_sub(app_model: 'AppModel') -> Sub:
    """Create a Sub that manages the Unix socket server lifecycle."""

    def start(send: Callable[[Msg], None]) -> Callable[[], None]:
        _cleanup_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        server.settimeout(0.5)

        print(f"viewer socket: {SOCKET_PATH}", file=sys.stderr)

        running = threading.Event()
        running.set()

        def serve():
            while running.is_set():
                try:
                    conn, _ = server.accept()
                    t = threading.Thread(
                        target=_handle_client,
                        args=(conn, send, app_model),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue
                except OSError:
                    break

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        def stop():
            running.clear()
            server.close()
            _cleanup_socket()

        return stop

    return Sub(key="socket-server", start=start)


# Legacy API — kept for backwards compatibility but apps should use subscriptions
def start_socket_server(program, app_model: 'AppModel') -> Callable[[], None]:
    """Start the Unix domain socket server on a background thread.

    Returns a callable that stops the server and cleans up.
    Prefer using subscriptions instead.
    """
    sub = make_socket_sub(app_model)
    return sub.start(program.send)
