"""Centralized styles — powered by snaptui's AppTheme."""

from snaptui import AppThemeCharm, Style

theme = AppThemeCharm()

# ── From theme ──
TITLE_STYLE      = theme.title
SUBTITLE_STYLE   = theme.subtitle
HELP_STYLE       = theme.help
ERROR_STYLE      = theme.error
DIM_STYLE        = theme.help

HEADER_FOCUSED   = theme.section_focused
HEADER_NORMAL    = theme.section_blurred
FIELD_LABEL      = theme.field_label

BORDER_ACTIVE    = theme.border_active
BORDER_INACTIVE  = theme.border_inactive

OVERLAY_STYLE    = theme.overlay

ITEM_SELECTED    = theme.item_selected
ITEM_NORMAL      = theme.item_normal
ITEM_DESC        = theme.item_description

# ── App-specific ──
TEXT_STYLE       = Style().fg("#FAFAFA")
FIELD_VALUE      = Style().fg("#FAFAFA")
FILE_VIEWABLE    = Style().fg("#02BF87")
FILE_NORMAL      = Style().fg("#AFAFAF")
