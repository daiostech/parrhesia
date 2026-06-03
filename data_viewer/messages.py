"""Messages for screen navigation and socket communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScreenPush:
    """Push a new screen onto the stack."""
    screen: str  # 'pipeline', 'file_viewer'
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScreenPop:
    """Pop the current screen off the stack."""


@dataclass(frozen=True)
class SocketMsg:
    """Message received from the Unix socket."""
    action: str
    data: dict[str, Any] = field(default_factory=dict)
