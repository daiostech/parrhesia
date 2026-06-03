"""Screen 2: Pipeline View — step list + detail panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snaptui import (
    Cmd, KeyMsg, Msg, WindowSizeMsg,
    Style, join_horizontal, TOP,
)
from snaptui.components import List, Viewport
from snaptui.components.help import Help, KeyBinding
from snaptui.components.table import Table, Column

from data_viewer.messages import ScreenPush, ScreenPop
from data_viewer.models.manifest import RunManifest, Step
from data_viewer.models.file_data import load_yaml_text
from data_viewer.shared.styles import (
    TITLE_STYLE, HELP_STYLE, DIM_STYLE, FIELD_LABEL, FIELD_VALUE,
    HEADER_FOCUSED, HEADER_NORMAL, FILE_VIEWABLE, FILE_NORMAL,
    ITEM_SELECTED, ITEM_NORMAL, ITEM_DESC,
    BORDER_ACTIVE, BORDER_INACTIVE, OVERLAY_STYLE,
)
from data_viewer.shared.syntax import colorize_yaml, add_line_numbers


# ── Item types for left panel ──

@dataclass
class HeaderEntry:
    """Section header label — skipped during navigation."""
    label: str


@dataclass
class SeparatorEntry:
    """Visual divider — skipped during navigation."""
    pass


@dataclass
class ResultsEntry:
    """Run-level results view."""
    file_count: int = 0


@dataclass
class ManifestEntry:
    """Raw manifest YAML view."""
    pass


# Union type for items in the left panel list
PipelineItem = Step | HeaderEntry | SeparatorEntry | ResultsEntry | ManifestEntry


# ── Delegates ──

class PipelineDelegate:
    """Renders all item types in the left panel list."""

    def render(self, item: PipelineItem, width: int, selected: bool) -> str:
        if isinstance(item, HeaderEntry):
            return ITEM_NORMAL.render(item.label)
        if isinstance(item, SeparatorEntry):
            return DIM_STYLE.render("─" * max(1, width - 2))
        if isinstance(item, ResultsEntry):
            style = ITEM_SELECTED if selected else ITEM_NORMAL
            indicator = f"  [{item.file_count} files]" if item.file_count else ""
            return style.render(f"Results{indicator}")
        if isinstance(item, ManifestEntry):
            style = ITEM_SELECTED if selected else ITEM_NORMAL
            return style.render("Manifest")
        # Step
        style = ITEM_SELECTED if selected else ITEM_NORMAL
        files = item.viewable_files()
        file_indicator = f"  [{len(files)} files]" if files else ""
        return style.render(f"{item.display_name}{file_indicator}")

    def height(self, item: PipelineItem, width: int) -> int:
        return 1


class FileDelegate:
    """Renders file items in the file picker list."""

    def render(self, item: tuple[str, str], width: int, selected: bool) -> str:
        label, path = item
        if selected:
            return ITEM_SELECTED.render(f"{label}: {path}")
        else:
            return FILE_VIEWABLE.render(f"  {label}: {path}")

    def height(self, item: tuple[str, str], width: int) -> int:
        return 1


# ── Detail formatting ──

def _viewable_files_for_step(step: Step) -> list[tuple[str, str]]:
    """Get all viewable files from inputs and outputs."""
    files = []
    for label, path in step.inputs.items():
        if path.endswith((".jsonl", ".json", ".yaml", ".yml", ".md")):
            files.append((label, path))
    for label, path in step.outputs.items():
        if path.endswith((".jsonl", ".json", ".yaml", ".yml", ".md")):
            files.append((label, path))
    return files


def _kv_table(data: dict[str, Any]) -> str:
    """Format a dict as an aligned key-value table."""
    rows = []
    for k, v in data.items():
        if isinstance(v, float):
            rows.append([k, f"{v:.4f}"])
        elif isinstance(v, list):
            rows.append([k, ", ".join(str(x) for x in v)])
        else:
            rows.append([k, str(v)])
    table = Table(
        columns=[Column("Key"), Column("Value")],
        rows=rows,
        header_style=Style(),  # no header for inline tables
    )
    # Render without header — just the rows, indented
    widths = table._col_widths()
    lines = []
    from snaptui import strutil
    gap = "  "
    for row in rows:
        cells = []
        for i in range(2):
            val = row[i] if i < len(row) else ""
            cells.append(strutil.pad_right(val, widths[i]))
        lines.append("    " + gap.join(cells))
    return "\n".join(lines)


def _format_step_detail(step: Step, run_results: dict[str, Any],
                        detail_focused: bool, selected_file: int,
                        width: int) -> str:
    """Format step details with gig_crm-style section headers."""
    lines: list[str] = []

    # ── Info section ──
    lines.append(HEADER_NORMAL.render(f" {step.display_name} "))
    lines.append("")

    if step.command:
        lines.append("  " + FIELD_LABEL.render("Command:"))
        lines.append(f"    {step.command}")
        lines.append("")

    if step.hyperparameters:
        lines.append("  " + FIELD_LABEL.render("Hyperparameters:"))
        lines.append(_kv_table(step.hyperparameters))
        lines.append("")

    if step.metrics:
        lines.append("  " + FIELD_LABEL.render("Metrics:"))
        lines.append(_kv_table(step.metrics))
        lines.append("")

    if step.extra:
        lines.append("  " + FIELD_LABEL.render("Details:"))
        lines.append(_kv_table(step.extra))
        lines.append("")

    if step.notes:
        lines.append("  " + FIELD_LABEL.render("Notes:"))
        lines.append(f"    {step.notes}")
        lines.append("")

    if step.completed_at:
        lines.append("  " + FIELD_LABEL.render("Completed:") + f" {step.completed_at}")
        lines.append("")

    if step.hub_refs:
        lines.append("  " + FIELD_LABEL.render("Hub References:"))
        lines.append(_kv_table(step.hub_refs))
        lines.append("")

    if run_results:
        lines.append("  " + FIELD_LABEL.render("Run Results:"))
        lines.append(_kv_table(run_results))
        lines.append("")

    # ── Input Files section ──
    input_files = list(step.inputs.items())
    if input_files:
        lines.append(HEADER_NORMAL.render(" Input Files "))
        lines.append("")
        for label, path in input_files:
            viewable = path.endswith((".jsonl", ".json", ".yaml", ".yml", ".md"))
            lines.append(_format_file_line(label, path, viewable, detail_focused, selected_file))
            if viewable:
                selected_file -= 1
        lines.append("")

    # ── Output Files section ──
    output_files = list(step.outputs.items())
    if output_files:
        lines.append(HEADER_NORMAL.render(" Output Files "))
        lines.append("")
        for label, path in output_files:
            viewable = path.endswith((".jsonl", ".json", ".yaml", ".yml", ".md"))
            lines.append(_format_file_line(label, path, viewable, detail_focused, selected_file))
            if viewable:
                selected_file -= 1
        lines.append("")

    return "\n".join(lines)


def _format_file_line(label: str, path: str, viewable: bool,
                      detail_focused: bool, remaining_idx: int) -> str:
    """Format a single file line, with > cursor if selected."""
    if viewable and detail_focused and remaining_idx == 0:
        return "  " + ITEM_SELECTED.render(f" {label}: {path}")
    elif viewable:
        return "  " + FILE_VIEWABLE.render(f"  {label}: {path}")
    else:
        return "  " + FILE_NORMAL.render(f"  {label}: {path}")


def _format_results_detail(manifest: RunManifest, project_root: Path,
                           result_files: list[str],
                           detail_focused: bool, selected_file: int) -> str:
    """Format the Results detail pane — metrics + result files."""
    lines: list[str] = []
    lines.append(HEADER_NORMAL.render(" Results "))
    lines.append("")

    if manifest.results:
        lines.append("  " + FIELD_LABEL.render("Metrics:"))
        lines.append(_kv_table(manifest.results))
        lines.append("")

    if result_files:
        lines.append(HEADER_NORMAL.render(" Result Files "))
        lines.append("")
        for i, f in enumerate(result_files):
            if detail_focused and i == selected_file:
                lines.append("  " + ITEM_SELECTED.render(f" {Path(f).name}: {f}"))
            else:
                lines.append("  " + FILE_VIEWABLE.render(f"  {Path(f).name}: {f}"))
        lines.append("")
    elif not manifest.results:
        lines.append("  " + DIM_STYLE.render("(no results)"))
        lines.append("")

    return "\n".join(lines)


# ── Screen Model ──

class PipelineModel:
    """Screen model for the pipeline/steps view."""

    PANEL_STEPS = 0
    PANEL_DETAIL = 1
    PANEL_FILES = 2  # file picker overlay

    def __init__(self, manifest: RunManifest, project_root: Path):
        self.manifest = manifest
        self.project_root = project_root
        self.width = 80
        self.height = 24

        # Discover result files for the Results entry
        self._result_files = manifest.result_files(project_root)

        # Build unified items list
        # With steps: Manifest + separator + Steps: header + steps + separator + Results
        # Without steps: Manifest + separator + Results
        items: list[PipelineItem] = []
        items.append(ManifestEntry())
        items.append(SeparatorEntry())
        if manifest.steps:
            items.append(HeaderEntry("Steps:"))
            items.extend(manifest.steps)
            items.append(SeparatorEntry())
        items.append(ResultsEntry(file_count=len(self._result_files)))

        # Left panel: unified list
        self.step_list = List(delegate=PipelineDelegate(), spacing=0)
        self.step_list.set_items(items)
        self._items = items

        # Right panel: detail viewport
        self.detail_vp = Viewport()

        # File cursor in detail panel
        self.selected_file = 0
        self._viewable_files: list[tuple[str, str]] = []

        # Active panel
        self.panel = self.PANEL_STEPS

        # File picker overlay
        self.file_list: List | None = None

    def init(self) -> Cmd:
        # Start cursor on first selectable item (skip HeaderEntry)
        for i, item in enumerate(self._items):
            if not isinstance(item, (SeparatorEntry, HeaderEntry)):
                self.step_list.cursor = i
                self._on_item_selected(item)
                break
        return None

    def _on_item_selected(self, item: PipelineItem) -> None:
        """Handle selection of any item type."""
        if isinstance(item, Step):
            self._viewable_files = _viewable_files_for_step(item)
            self.selected_file = 0
            self._rebuild_detail()
            self.detail_vp.goto_top()
        elif isinstance(item, ResultsEntry):
            self._viewable_files = []
            self.selected_file = 0
            self._rebuild_detail()
            self.detail_vp.goto_top()
        elif isinstance(item, ManifestEntry):
            self._viewable_files = []
            self.selected_file = 0
            self._rebuild_detail()
            self.detail_vp.goto_top()

    def _rebuild_detail(self) -> None:
        """Rebuild the detail viewport content with current cursor state."""
        detail_focused = self.panel == self.PANEL_DETAIL
        item = self._current_item()
        if item is None:
            return

        if isinstance(item, Step):
            detail_width = max(20, int(self.width * 0.65) - 4)
            text = _format_step_detail(
                item, self.manifest.results,
                detail_focused, self.selected_file, detail_width,
            )
            self.detail_vp.set_content(text)
        elif isinstance(item, ResultsEntry):
            text = _format_results_detail(
                self.manifest, self.project_root, self._result_files,
                detail_focused, self.selected_file,
            )
            self.detail_vp.set_content(text)
        elif isinstance(item, ManifestEntry):
            self._load_manifest_yaml()

    def _load_manifest_yaml(self) -> None:
        """Load and colorize the raw manifest YAML."""
        manifest_path = self.manifest.manifest_path
        if not manifest_path or not Path(manifest_path).exists():
            self.detail_vp.set_content(DIM_STYLE.render("(manifest file not found)"))
            return
        try:
            text = load_yaml_text(manifest_path)
            text = colorize_yaml(text)
            text = add_line_numbers(text)
            self.detail_vp.set_content(text)
        except Exception as e:
            self.detail_vp.set_content(f"Error loading manifest: {e}")

    def _current_item(self) -> PipelineItem | None:
        return self.step_list.selected_item()

    def _current_step(self) -> Step | None:
        item = self._current_item()
        return item if isinstance(item, Step) else None

    def _file_count(self) -> int:
        """Number of selectable files in current view."""
        item = self._current_item()
        if isinstance(item, Step):
            return len(self._viewable_files)
        elif isinstance(item, ResultsEntry):
            return len(self._result_files)
        return 0

    def _selected_file_path(self) -> str | None:
        """Get the path of the currently selected file."""
        item = self._current_item()
        if isinstance(item, Step):
            if 0 <= self.selected_file < len(self._viewable_files):
                return self._viewable_files[self.selected_file][1]
        elif isinstance(item, ResultsEntry):
            if 0 <= self.selected_file < len(self._result_files):
                return self._result_files[self.selected_file]
        return None

    def _open_file_picker(self, files: list[tuple[str, str]]) -> None:
        self.file_list = List(delegate=FileDelegate(), spacing=0)
        self.file_list.set_items(files)
        self.file_list.width = min(70, self.width - 10)
        self.file_list.height = min(len(files) + 2, self.height - 6)
        self.panel = self.PANEL_FILES

    def _skip_non_selectable(self, direction: int) -> None:
        """If cursor is on a non-selectable entry, keep nudging until it lands on a selectable one."""
        items = self._items
        if not items:
            return
        cursor = self.step_list.cursor
        while 0 <= cursor < len(items) and isinstance(items[cursor], (SeparatorEntry, HeaderEntry)):
            next_cursor = cursor + direction
            if 0 <= next_cursor < len(items):
                cursor = next_cursor
            else:
                # Hit the edge — reverse direction to find nearest selectable
                cursor = self.step_list.cursor
                direction = -direction
                while 0 <= cursor < len(items) and isinstance(items[cursor], (SeparatorEntry, HeaderEntry)):
                    cursor += direction
                break
        if 0 <= cursor < len(items):
            self.step_list.cursor = cursor

    def update(self, msg: Msg) -> tuple[PipelineModel, Cmd]:
        if isinstance(msg, WindowSizeMsg):
            self.width = msg.width
            self.height = msg.height
            left_w = max(20, int(msg.width * 0.35) - 4)
            right_w = max(20, int(msg.width * 0.65) - 4)
            self.step_list.width = left_w
            self.step_list.height = max(5, msg.height - 6)
            self.detail_vp.width = right_w
            self.detail_vp.height = max(5, msg.height - 6)
            if self.file_list:
                self.file_list.width = min(70, msg.width - 10)
            return self, None

        if isinstance(msg, KeyMsg):
            # File picker overlay takes priority
            if self.panel == self.PANEL_FILES and self.file_list:
                return self._update_file_picker(msg)

            if msg.key in ('q', 'esc'):
                return self, ScreenPop

            if msg.key == 'tab':
                item = self._current_item()
                if isinstance(item, (Step, ResultsEntry, ManifestEntry)):
                    self.panel = self.PANEL_DETAIL if self.panel == self.PANEL_STEPS else self.PANEL_STEPS
                    self._rebuild_detail()
                return self, None

            # Right arrow: left panel → detail panel
            if msg.key == 'right' and self.panel == self.PANEL_STEPS:
                item = self._current_item()
                if isinstance(item, (Step, ResultsEntry, ManifestEntry)):
                    self.panel = self.PANEL_DETAIL
                    self._rebuild_detail()
                return self, None

            # Left arrow: detail → left panel, left panel → back to run list
            if msg.key == 'left':
                if self.panel == self.PANEL_DETAIL:
                    self.panel = self.PANEL_STEPS
                    self._rebuild_detail()
                    return self, None
                elif self.panel == self.PANEL_STEPS:
                    return self, ScreenPop

            if self.panel == self.PANEL_STEPS:
                return self._update_steps(msg)
            elif self.panel == self.PANEL_DETAIL:
                return self._update_detail_keys(msg)

        return self, None

    def _update_steps(self, msg: KeyMsg) -> tuple[PipelineModel, Cmd]:
        old_cursor = self.step_list.cursor
        self.step_list, cmd = self.step_list.update(msg)

        # Skip separators
        if self.step_list.cursor != old_cursor:
            direction = 1 if self.step_list.cursor > old_cursor else -1
            self._skip_non_selectable(direction)

        if self.step_list.cursor != old_cursor:
            item = self._current_item()
            if item and not isinstance(item, SeparatorEntry):
                self._on_item_selected(item)

        if msg.key == 'enter':
            item = self._current_item()
            if isinstance(item, Step):
                files = item.viewable_files()
                if len(files) == 1:
                    _, path = files[0]
                    return self, self._make_open_file_cmd(path)
                elif len(files) > 1:
                    self._open_file_picker(files)
            elif isinstance(item, ResultsEntry):
                if self._result_files:
                    # Enter results detail view via tab
                    self.panel = self.PANEL_DETAIL
                    self.selected_file = 0
                    self._rebuild_detail()
            elif isinstance(item, ManifestEntry):
                # Enter manifest detail view via tab
                self.panel = self.PANEL_DETAIL
                self._rebuild_detail()

        return self, cmd

    def _update_detail_keys(self, msg: KeyMsg) -> tuple[PipelineModel, Cmd]:
        item = self._current_item()

        # ManifestEntry: j/k scrolls the YAML viewport
        if isinstance(item, ManifestEntry):
            if msg.key in ('j', 'down'):
                self.detail_vp.line_down(3)
                return self, None
            if msg.key in ('k', 'up'):
                self.detail_vp.line_up(3)
                return self, None
            self.detail_vp, cmd = self.detail_vp.update(msg)
            return self, cmd

        # Step and ResultsEntry: j/k navigates file cursor
        fc = self._file_count()

        if msg.key in ('j', 'down'):
            if fc > 0 and self.selected_file < fc - 1:
                self.selected_file += 1
                self._rebuild_detail()
            elif fc == 0:
                self.detail_vp, _ = self.detail_vp.update(msg)
            return self, None

        if msg.key in ('k', 'up'):
            if fc > 0 and self.selected_file > 0:
                self.selected_file -= 1
                self._rebuild_detail()
            elif fc == 0:
                self.detail_vp, _ = self.detail_vp.update(msg)
            return self, None

        if msg.key == 'enter':
            path = self._selected_file_path()
            if path:
                return self, self._make_open_file_cmd(path)
            return self, None

        # Other keys (pgup, pgdown, etc.) go to viewport
        self.detail_vp, cmd = self.detail_vp.update(msg)
        return self, cmd

    def _update_file_picker(self, msg: KeyMsg) -> tuple[PipelineModel, Cmd]:
        if msg.key in ('q', 'esc'):
            self.file_list = None
            self.panel = self.PANEL_STEPS
            return self, None

        if msg.key == 'enter':
            selected = self.file_list.selected_item()
            if selected:
                _, path = selected
                self.file_list = None
                self.panel = self.PANEL_STEPS
                return self, self._make_open_file_cmd(path)
            return self, None

        self.file_list, cmd = self.file_list.update(msg)
        return self, cmd

    def _make_open_file_cmd(self, rel_path: str) -> Cmd:
        abs_path = str(self.project_root / rel_path)
        return lambda: ScreenPush('file_viewer', {
            'abs_path': abs_path,
            'rel_path': rel_path,
        })

    def view(self) -> str:
        # Header
        header = TITLE_STYLE.render(
            f"{self.manifest.run_id}  |  Approach {self.manifest.approach}  |  {self.manifest.base_model}"
        )

        # Layout widths
        total_w = self.width - 2
        left_w = max(20, int(total_w * 0.35))
        right_w = max(20, total_w - left_w)

        # Left panel
        left_style = (BORDER_ACTIVE if self.panel == self.PANEL_STEPS else BORDER_INACTIVE).width(left_w)
        left_panel = left_style.render(self.step_list.view())

        # Right panel
        right_style = (BORDER_ACTIVE if self.panel == self.PANEL_DETAIL else BORDER_INACTIVE).width(right_w)
        right_panel = right_style.render(self.detail_vp.view())

        # Join panels
        panels = join_horizontal(TOP, left_panel, right_panel)

        # Help
        item = self._current_item()
        if self.panel == self.PANEL_STEPS:
            help = Help(bindings=[
                KeyBinding("\u2191/\u2193", "navigate"),
                KeyBinding("\u2192", "detail"),
                KeyBinding("enter", "open"),
                KeyBinding("\u2190", "back"),
            ], width=self.width)
        elif isinstance(item, ManifestEntry):
            help = Help(bindings=[
                KeyBinding("\u2191/\u2193", "scroll"),
                KeyBinding("\u2190", "back"),
            ], width=self.width)
        else:
            help = Help(bindings=[
                KeyBinding("\u2191/\u2193", "select file"),
                KeyBinding("enter", "open"),
                KeyBinding("\u2190", "back"),
            ], width=self.width)
        help_text = HELP_STYLE.render(help.short_help())

        body = f"{header}\n{panels}\n{help_text}"

        # File picker overlay
        if self.file_list and self.panel == self.PANEL_FILES:
            overlay_header = TITLE_STYLE.render(" Select a file ")
            overlay_content = self.file_list.view()
            overlay_help_w = Help(bindings=[
                KeyBinding("\u2191/\u2193", "navigate"),
                KeyBinding("enter", "open"),
                KeyBinding("esc", "cancel"),
            ], width=self.width)
            overlay_help = HELP_STYLE.render(overlay_help_w.short_help())
            overlay = OVERLAY_STYLE.render(f"{overlay_header}\n\n{overlay_content}\n\n{overlay_help}")
            body = f"{header}\n{panels}\n{overlay}\n{help_text}"

        return body
