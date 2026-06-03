"""Shared syntax highlighting — VS Code Dark+ palette for JSON & YAML."""

from __future__ import annotations

import re

# ── ANSI color constants (VS Code Dark+ palette) ──
ANSI_RE = re.compile(r'\033\[[^m]*m')
_KEY = "\033[1;38;2;156;220;254m"      # #9CDCFE bold - keys (light blue)
_STR = "\033[1;38;2;206;145;120m"      # #CE9178 bold - string values (orange)
_NUM = "\033[1;38;2;181;206;168m"      # #B5CEA8 bold - numbers (light green)
_KW = "\033[1;38;2;86;156;214m"        # #569CD6 bold - true/false/null (blue)
_COMMENT = "\033[1;38;2;106;153;85m"   # #6A9955 bold - comments (green)
_LINE_NUM = "\033[38;2;133;133;133m"   # #858585 - line numbers (dim gray)
_RESET = "\033[0m"

GUTTER_WIDTH = 6

# ── JSON tokenizer ──
_JSON_TOKEN_RE = re.compile(
    r'("(?:[^"\\]|\\.)*")(\s*:)?'           # quoted string, optionally key
    r'|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'  # number
    r'|\b(true|false|null)\b'                # keywords
    r'|(//[^\n]*)'                           # // comment
    r'|(#[^\n]*)'                            # # comment
)

# ── YAML tokenizer ──
_YAML_TOKEN_RE = re.compile(
    r'(#[^\n]*)'                                          # comment
    r'|^([ \t]*(?:-\s+)?[\w][\w.\-]*\s*)(:)'             # key: (at line start, optional list prefix)
    r"|('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"          # quoted strings
    r'|(?<=:\s)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s*(?:#|$))'  # number after colon
    r'|\b(true|false|null|yes|no|True|False|Null|Yes|No)\b'        # keywords
    , re.MULTILINE
)


def visible_len(s: str) -> int:
    """Length of string excluding ANSI escape sequences."""
    return len(ANSI_RE.sub('', s))


def colorize_json(text: str) -> str:
    """Add VS Code Dark+ syntax highlighting to JSON text."""
    def replacer(m):
        if m.group(1):  # quoted string
            if m.group(2):  # followed by colon -> key
                return f'{_KEY}{m.group(1)}{_RESET}{m.group(2)}'
            return f'{_STR}{m.group(1)}{_RESET}'
        if m.group(3):  # number
            return f'{_NUM}{m.group(3)}{_RESET}'
        if m.group(4):  # keyword
            return f'{_KW}{m.group(4)}{_RESET}'
        if m.group(5):  # // comment
            return f'{_COMMENT}{m.group(5)}{_RESET}'
        if m.group(6):  # # comment
            return f'{_COMMENT}{m.group(6)}{_RESET}'
        return m.group(0)
    return _JSON_TOKEN_RE.sub(replacer, text)


def colorize_yaml(text: str) -> str:
    """Add VS Code Dark+ syntax highlighting to YAML text."""
    def replacer(m):
        if m.group(1):  # comment
            return f'{_COMMENT}{m.group(1)}{_RESET}'
        if m.group(2):  # key name
            return f'{_KEY}{m.group(2)}{_RESET}{m.group(3)}'
        if m.group(4):  # quoted string
            return f'{_STR}{m.group(4)}{_RESET}'
        if m.group(5):  # number
            return f'{_NUM}{m.group(5)}{_RESET}'
        if m.group(6):  # keyword
            return f'{_KW}{m.group(6)}{_RESET}'
        return m.group(0)
    return _YAML_TOKEN_RE.sub(replacer, text)


def add_line_numbers(text: str, start: int = 1) -> str:
    """Add sequential line numbers to text."""
    lines = text.split("\n")
    max_num = start + len(lines) - 1
    width = max(GUTTER_WIDTH, len(str(max_num)) + 2)
    result = []
    for i, line in enumerate(lines):
        num = start + i
        result.append(f"{_LINE_NUM}{num:>{width - 1}} {_RESET}{line}")
    return "\n".join(result)


def add_jsonl_gutter(header: str, text: str, line_num: int, gutter_width: int) -> str:
    """Add line number gutter to a JSONL record block.

    Header gets blank gutter + comment color, first JSON line gets the
    original line number, continuation lines get blank gutter.
    """
    blank = " " * gutter_width
    lines = text.split("\n")
    result = [f"{blank}{_COMMENT}{header}{_RESET}"]
    for j, line in enumerate(lines):
        if j == 0:
            result.append(f"{_LINE_NUM}{line_num:>{gutter_width - 1}} {_RESET}{line}")
        else:
            result.append(f"{blank}{line}")
    return "\n".join(result)
