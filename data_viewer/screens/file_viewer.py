"""Screen 3: File Viewer — JSONL pagination, JSON/YAML/text display."""

from __future__ import annotations

import json
from pathlib import Path

from snaptui import Cmd, KeyMsg, Msg, WindowSizeMsg
from snaptui.components import Viewport
from snaptui.components.help import Help, KeyBinding
from snaptui.components.progress import Progress

from data_viewer.messages import ScreenPop
from data_viewer.models.file_data import JsonlReader, load_yaml_text, format_file_size
from data_viewer.shared.styles import TITLE_STYLE, HELP_STYLE, BORDER_INACTIVE, DIM_STYLE
from data_viewer.shared.syntax import (
    visible_len as _visible_len,
    colorize_json as _colorize_json,
    colorize_yaml as _colorize_yaml,
    add_line_numbers as _add_line_numbers,
    add_jsonl_gutter as _add_jsonl_gutter,
    GUTTER_WIDTH,
)

MAX_RENDER_CHARS = 200_000
MAX_RENDER_LINES = 5_000


def _truncate(text: str) -> tuple[str, bool]:
    lines = text.split("\n")
    if len(lines) > MAX_RENDER_LINES:
        return "\n".join(lines[:MAX_RENDER_LINES]), True
    if len(text) > MAX_RENDER_CHARS:
        return text[:MAX_RENDER_CHARS], True
    return text, False


def _load_json_pretty(path: str) -> str:
    file_size = Path(path).stat().st_size
    with open(path) as f:
        data = json.load(f)

    if file_size < 200_000:
        return json.dumps(data, indent=2, ensure_ascii=False)

    if isinstance(data, list):
        if len(data) <= 20:
            return json.dumps(data, indent=2, ensure_ascii=False)
        header = f"# Array with {len(data)} records\n\n"
        body = json.dumps(data[:10], indent=2, ensure_ascii=False)
        return header + body + f"\n\n# ... ({len(data) - 10} more records not shown)"
    elif isinstance(data, dict):
        parts = ["{"]
        keys = list(data.keys())
        for i, k in enumerate(keys):
            v = data[k]
            comma = "," if i < len(keys) - 1 else ""
            if isinstance(v, list) and len(v) > 5:
                items_str = json.dumps(v[:3], indent=4, ensure_ascii=False)
                parts.append(f'  "{k}": [  // {len(v)} items total')
                for line in items_str.strip("[]").strip().split("\n"):
                    parts.append(f"  {line}")
                parts.append(f"    // ... {len(v) - 3} more items")
                parts.append(f"  ]{comma}")
            else:
                val_str = json.dumps(v, indent=2, ensure_ascii=False)
                if "\n" in val_str:
                    parts.append(f'  "{k}": {val_str.replace(chr(10), chr(10) + "  ")}{comma}')
                else:
                    parts.append(f'  "{k}": {val_str}{comma}')
        parts.append("}")
        return "\n".join(parts)
    else:
        return json.dumps(data, indent=2, ensure_ascii=False)




class FileViewerModel:
    """Screen model for viewing file contents."""

    def __init__(self, abs_path: str, rel_path: str):
        self.abs_path = abs_path
        self.rel_path = rel_path
        self.viewport = Viewport()
        self.width = 80
        self.height = 24
        self._reader: JsonlReader | None = None
        self._is_jsonl = abs_path.endswith(".jsonl")
        self._page_start = 0
        self._page_end = 0
        self._header = ""
        self._content_loaded = False
        self._oversized = False  # True when single record exceeds viewport

    def init(self) -> Cmd:
        self._load_content()
        return None

    def _load_content(self) -> None:
        path = Path(self.abs_path)
        if not path.exists():
            self._header = f" {self.rel_path}  |  NOT FOUND"
            self.viewport.set_content(f"File not found: {self.abs_path}")
            self._content_loaded = True
            return

        size = format_file_size(path.stat().st_size)

        if self._is_jsonl:
            self._reader = JsonlReader(self.abs_path)
            if self._reader.count == 0:
                self._header = f" {self.rel_path}  |  {size}  |  (empty)"
                self.viewport.set_content("(empty JSONL file)")
            else:
                self._show_page(0)
        elif self.abs_path.endswith(".json"):
            self._header = f" {self.rel_path}  |  {size}"
            try:
                text = _load_json_pretty(self.abs_path)
                text = _colorize_json(text)
                self.viewport.set_content(_add_line_numbers(text))
            except Exception as e:
                self.viewport.set_content(f"Error loading JSON: {e}")
        elif self.abs_path.endswith((".yaml", ".yml")):
            self._header = f" {self.rel_path}  |  {size}"
            try:
                text, _ = _truncate(load_yaml_text(self.abs_path))
                text = _colorize_yaml(text)
                self.viewport.set_content(_add_line_numbers(text))
            except Exception as e:
                self.viewport.set_content(f"Error loading YAML: {e}")
        else:
            self._header = f" {self.rel_path}  |  {size}"
            try:
                with open(self.abs_path) as f:
                    text = f.read(MAX_RENDER_CHARS)
                self.viewport.set_content(_add_line_numbers(text))
            except Exception as e:
                self.viewport.set_content(f"Error: {e}")

        self._content_loaded = True

    def _content_height(self) -> int:
        """Available lines for records (viewport height)."""
        return max(5, self.viewport.height)

    def _wrapped_line_count(self, text: str) -> int:
        """Count rendered lines after wrapping to viewport width."""
        width = max(20, self.viewport.width)
        count = 0
        for line in text.split("\n"):
            visible_len = _visible_len(line)
            if visible_len <= width or width <= 0:
                count += 1
            else:
                count += (visible_len + width - 1) // width
        return count

    def _gutter_width(self) -> int:
        """Compute gutter width based on max line number in the file."""
        if not self._reader or self._reader.count == 0:
            return GUTTER_WIDTH
        max_line = self._reader.get_line_number(self._reader.count - 1)
        return max(GUTTER_WIDTH, len(str(max_line)) + 2)

    def _show_page(self, start: int) -> None:
        """Load records that fit on screen starting at the given index."""
        if not self._reader:
            return

        total = self._reader.count
        start = max(0, min(start, total - 1))
        budget = self._content_height()
        gw = self._gutter_width()

        blocks = []
        lines_used = 0
        i = start
        while i < total:
            try:
                text = self._reader.get_record_raw(i)
            except Exception as e:
                text = f"Error: {e}"
            text = _colorize_json(text)
            header = f"// --- record {i + 1} ---"
            line_num = self._reader.get_line_number(i)
            block = _add_jsonl_gutter(header, text, line_num, gw)
            block_lines = self._wrapped_line_count(block)

            if blocks and lines_used + block_lines > budget:
                break

            blocks.append(block)
            lines_used += block_lines
            i += 1

        self._page_start = start
        self._page_end = start + len(blocks)
        self._oversized = len(blocks) == 1 and lines_used > budget

        size = format_file_size(self._reader.file_size)
        self._header = (
            f" {self.rel_path}  |  {size}  |  "
            f"Records {self._page_start + 1}-{self._page_end}/{total}  |  "
            f"\u2190 \u2192 to navigate"
        )

        self.viewport.set_content("\n".join(blocks))
        self.viewport.goto_top()

    def _show_page_backwards(self, end: int) -> None:
        """Find the page that ends just before `end`, fitting to screen."""
        if not self._reader:
            return

        budget = self._content_height()
        gw = self._gutter_width()
        i = end - 1
        lines_used = 0
        first = i
        while i >= 0:
            try:
                text = self._reader.get_record_raw(i)
            except Exception:
                text = "{}"
            header = f"// --- record {i + 1} ---"
            line_num = self._reader.get_line_number(i)
            block = _add_jsonl_gutter(header, text, line_num, gw)
            block_lines = self._wrapped_line_count(block)

            if i < end - 1 and lines_used + block_lines > budget:
                break
            lines_used += block_lines
            first = i
            i -= 1

        self._show_page(first)

    def update(self, msg: Msg) -> tuple[FileViewerModel, Cmd]:
        if isinstance(msg, WindowSizeMsg):
            self.width = msg.width
            self.height = msg.height
            content_width = max(20, msg.width - 4)
            content_height = max(5, msg.height - 6)
            self.viewport.width = content_width
            self.viewport.height = content_height
            # Recompute page on resize for JSONL
            if self._is_jsonl and self._reader and self._reader.count > 0:
                self._show_page(self._page_start)
            return self, None

        if isinstance(msg, KeyMsg):
            if msg.key in ('q', 'esc'):
                return self, ScreenPop
            if self._is_jsonl and self._reader:
                if msg.key == 'left':
                    if self._page_start > 0:
                        self._show_page_backwards(self._page_start)
                    return self, None
                if msg.key == 'right':
                    if self._page_end < self._reader.count:
                        self._show_page(self._page_end)
                    return self, None
                # Only allow scrolling for oversized single records
                if self._oversized:
                    self.viewport, cmd = self.viewport.update(msg)
                    return self, cmd
                return self, None
            # Non-JSONL: j/k scroll fast, left/right page
            if msg.key in ('j', 'down'):
                self.viewport.line_down(5)
                return self, None
            if msg.key in ('k', 'up'):
                self.viewport.line_up(5)
                return self, None
            if msg.key == 'right':
                self.viewport.page_down()
                return self, None
            if msg.key == 'left':
                self.viewport.page_up()
                return self, None
            self.viewport, cmd = self.viewport.update(msg)
            return self, cmd

        return self, None

    def view(self) -> str:
        border_style = BORDER_INACTIVE.width(self.width - 2)

        header = TITLE_STYLE.render(self._header)
        content = self.viewport.view()

        if self._is_jsonl and self._reader and self._reader.count > 0:
            if self._oversized:
                help = Help(bindings=[
                    KeyBinding("\u2191/\u2193", "scroll"),
                    KeyBinding("\u2190/\u2192", "prev/next page"),
                    KeyBinding("esc", "back"),
                ], width=self.width)
            else:
                help = Help(bindings=[
                    KeyBinding("\u2190/\u2192", "prev/next page"),
                    KeyBinding("esc", "back"),
                ], width=self.width)
        else:
            help = Help(bindings=[
                KeyBinding("\u2191/\u2193", "scroll"),
                KeyBinding("\u2190/\u2192", "page up/down"),
                KeyBinding("esc", "back"),
            ], width=self.width)
        help_text = HELP_STYLE.render(help.short_help())

        # Progress bar: pagination for JSONL, scroll position for others
        bar_width = max(10, self.width - 6)
        if self._is_jsonl and self._reader and self._reader.count > 0:
            pct = self._page_end / self._reader.count
            bar = Progress(percent=pct, width=bar_width, show_percent=False,
                           fill_char="━", empty_char="━")
        elif self.viewport.total_lines > self.viewport.height:
            bar = Progress(percent=self.viewport.scroll_percent, width=bar_width,
                           show_percent=False, fill_char="━", empty_char="━")
        else:
            bar = None

        parts = [header, content]
        if bar:
            parts.append(bar.view())
        parts.append(help_text)
        return border_style.render("\n".join(parts))
