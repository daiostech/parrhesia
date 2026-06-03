"""Screen 1: Run Selector — list of discovered runs."""

from __future__ import annotations

from snaptui import Cmd, KeyMsg, Msg, WindowSizeMsg, strutil
from snaptui.components import List
from snaptui.components.help import Help, KeyBinding

from data_viewer.messages import ScreenPush, ScreenPop
from data_viewer.models.manifest import RunManifest, discover_runs
from data_viewer.shared.styles import (
    TITLE_STYLE, SUBTITLE_STYLE, HELP_STYLE,
    ITEM_SELECTED, ITEM_NORMAL, ITEM_DESC,
    BORDER_INACTIVE,
)


class RunDelegate:
    """Renders run items in the list."""

    def render(self, item: RunManifest, width: int, selected: bool) -> str:
        style = ITEM_SELECTED if selected else ITEM_NORMAL
        desc_style = ITEM_DESC

        run_id = item.run_id
        approach = f"[{item.approach}]"
        model = item.base_model
        steps = f"{item.num_steps} steps"
        desc = item.description
        max_desc = width - 50
        if max_desc > 0:
            desc = strutil.truncate(desc, max_desc, "\u2026")

        line1 = style.render(f"{run_id}  {approach}  {model}  {steps}")
        line2 = desc_style.render(f"  {desc}")
        return f"{line1}\n{line2}"

    def height(self, item: RunManifest, width: int) -> int:
        return 2


class RunSelectorModel:
    """Screen model for selecting a run."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.runs: list[RunManifest] = []
        self.list = List(delegate=RunDelegate(), spacing=1)
        self.width = 80
        self.height = 24

    def init(self) -> Cmd:
        self.runs = discover_runs(self.project_root)
        self.list.set_items(self.runs)
        return None

    def update(self, msg: Msg) -> tuple[RunSelectorModel, Cmd]:
        if isinstance(msg, WindowSizeMsg):
            self.width = msg.width
            self.height = msg.height
            self.list.width = msg.width - 4
            self.list.height = msg.height - 4
            return self, None

        if isinstance(msg, KeyMsg):
            if msg.key in ('q', 'esc', 'ctrl+c'):
                return self, ScreenPop
            if msg.key in ('enter', 'right'):
                selected = self.list.selected_item()
                if selected:
                    return self, lambda: ScreenPush('pipeline', {'run': selected})
            self.list, cmd = self.list.update(msg)
            return self, cmd

        return self, None

    def view(self) -> str:
        border_style = BORDER_INACTIVE.width(self.width - 2)

        title = TITLE_STYLE.render("Parrhesia Data Viewer")
        subtitle = SUBTITLE_STYLE.render(f"  {len(self.runs)} runs discovered")
        header = f"{title} {subtitle}"

        list_view = self.list.view()
        pager = self.list.pager_view()

        help = Help(bindings=[
            KeyBinding("\u2191/\u2193", "navigate"),
            KeyBinding("\u2192/enter", "open"),
            KeyBinding("q", "quit"),
        ], width=self.width)
        help_text = HELP_STYLE.render(help.short_help())
        if pager:
            help_text = HELP_STYLE.render(f"{help.short_help()}  {pager}")

        body = f"{header}\n\n{list_view}\n\n{help_text}"
        return border_style.render(body)
