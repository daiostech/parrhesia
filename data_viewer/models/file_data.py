"""File loaders for JSONL (lazy paginated), JSON, and YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlReader:
    """Random-access JSONL reader using a byte-offset index.

    On open, scans for newlines to build offsets (~0.1s for 57MB).
    Then get_record(i) does seek() + readline() for O(1) access.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets: list[int] = []
        self._line_numbers: list[int] = []  # 1-based original line number per record
        self._file = None
        self._build_index()

    def _build_index(self) -> None:
        self._offsets = []
        self._line_numbers = []
        with open(self.path, "rb") as f:
            offset = 0
            line_num = 0
            for line in f:
                line_num += 1
                stripped = line.strip()
                if stripped:
                    self._offsets.append(offset)
                    self._line_numbers.append(line_num)
                offset += len(line)

    @property
    def count(self) -> int:
        return len(self._offsets)

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size

    def get_line_number(self, index: int) -> int:
        """Get the original 1-based line number for record at index."""
        if index < 0 or index >= self.count:
            raise IndexError(f"Record {index} out of range (0-{self.count - 1})")
        return self._line_numbers[index]

    def get_record(self, index: int) -> dict[str, Any]:
        """Get a single record by index (0-based). Returns parsed JSON."""
        if index < 0 or index >= self.count:
            raise IndexError(f"Record {index} out of range (0-{self.count - 1})")
        with open(self.path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline()
            return json.loads(line)

    def get_record_raw(self, index: int) -> str:
        """Get a single record as a pretty-printed JSON string."""
        record = self.get_record(index)
        return json.dumps(record, indent=2, ensure_ascii=False)


def load_yaml_text(path: str | Path) -> str:
    """Load a YAML file and return it as formatted text."""
    path = Path(path)
    with open(path) as f:
        return f.read()


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
