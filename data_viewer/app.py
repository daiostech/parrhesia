"""Main app model — screen stack navigation with Elm Architecture."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

from snaptui import Cmd, Msg, WindowSizeMsg, quit_cmd
from snaptui.model import Sub

from data_viewer.messages import ScreenPush, ScreenPop, SocketMsg
from data_viewer.screens.run_selector import RunSelectorModel
from data_viewer.screens.pipeline import PipelineModel
from data_viewer.screens.file_viewer import FileViewerModel


class Screen(Enum):
    RUN_SELECTOR = auto()
    PIPELINE = auto()
    FILE_VIEWER = auto()


class AppModel:
    """Top-level model managing a screen stack."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.screen = Screen.RUN_SELECTOR
        self._screen_stack: list[tuple[Screen, object]] = []
        self.width = 80
        self.height = 24

        # Active screen models
        self.run_selector = RunSelectorModel(project_root)
        self.pipeline: PipelineModel | None = None
        self.file_viewer: FileViewerModel | None = None

    def _active_model(self):
        """Return the currently active screen model."""
        if self.screen == Screen.RUN_SELECTOR:
            return self.run_selector
        elif self.screen == Screen.PIPELINE:
            return self.pipeline
        elif self.screen == Screen.FILE_VIEWER:
            return self.file_viewer

    def init(self) -> Cmd:
        return self.run_selector.init()

    def subscriptions(self) -> list[Sub]:
        from data_viewer.socket_server import make_socket_sub
        return [make_socket_sub(self)]

    def update(self, msg: Msg) -> tuple[AppModel, Cmd]:
        if isinstance(msg, WindowSizeMsg):
            self.width = msg.width
            self.height = msg.height
            # Forward to active screen
            active = self._active_model()
            if active:
                active.update(msg)
            return self, None

        # Handle screen navigation messages (returned as Cmd results)
        if isinstance(msg, ScreenPush):
            return self._handle_push(msg)

        if isinstance(msg, ScreenPop) or msg is ScreenPop:
            return self._handle_pop()

        # Handle socket messages
        if isinstance(msg, SocketMsg):
            return self._handle_socket(msg)

        # Delegate to active screen
        active = self._active_model()
        if active:
            _, cmd = active.update(msg)
            return self, cmd

        return self, None

    def _handle_push(self, msg: ScreenPush) -> tuple[AppModel, Cmd]:
        """Push a new screen onto the stack."""
        # Save current screen state
        self._screen_stack.append((self.screen, self._active_model()))

        if msg.screen == 'pipeline':
            run = msg.kwargs.get('run')
            self.pipeline = PipelineModel(run, self.project_root)
            self.pipeline.width = self.width
            self.pipeline.height = self.height
            # Forward current size
            self.pipeline.update(WindowSizeMsg(self.width, self.height))
            self.screen = Screen.PIPELINE
            return self, self.pipeline.init()

        elif msg.screen == 'file_viewer':
            abs_path = msg.kwargs.get('abs_path', '')
            rel_path = msg.kwargs.get('rel_path', '')
            self.file_viewer = FileViewerModel(abs_path, rel_path)
            self.file_viewer.width = self.width
            self.file_viewer.height = self.height
            self.file_viewer.update(WindowSizeMsg(self.width, self.height))
            self.screen = Screen.FILE_VIEWER
            return self, self.file_viewer.init()

        return self, None

    def _handle_pop(self) -> tuple[AppModel, Cmd]:
        """Pop the current screen off the stack."""
        if not self._screen_stack:
            # At top level — quit
            return self, quit_cmd
        prev_screen, prev_model = self._screen_stack.pop()
        self.screen = prev_screen

        # Restore and resize the previous screen
        if prev_screen == Screen.RUN_SELECTOR:
            self.run_selector = prev_model
            self.run_selector.update(WindowSizeMsg(self.width, self.height))
        elif prev_screen == Screen.PIPELINE:
            self.pipeline = prev_model
            self.pipeline.update(WindowSizeMsg(self.width, self.height))
        elif prev_screen == Screen.FILE_VIEWER:
            self.file_viewer = prev_model
            self.file_viewer.update(WindowSizeMsg(self.width, self.height))

        return self, None

    def _handle_socket(self, msg: SocketMsg) -> tuple[AppModel, Cmd]:
        """Handle messages from the Unix socket."""
        if msg.action == 'navigate':
            run_id = msg.data.get('run_id')
            step_name = msg.data.get('step')
            if run_id:
                # Find the run
                for run in self.run_selector.runs:
                    if run.run_id == run_id:
                        push = ScreenPush('pipeline', {'run': run})
                        return self._handle_push(push)
            return self, None

        return self, None

    def get_status(self) -> dict:
        """Return current viewer state for socket status queries."""
        status = {'screen': self.screen.name}
        if self.screen == Screen.PIPELINE and self.pipeline:
            status['run_id'] = self.pipeline.manifest.run_id
            step = self.pipeline._current_step()
            if step:
                status['step'] = step.name
        elif self.screen == Screen.FILE_VIEWER and self.file_viewer:
            status['file'] = self.file_viewer.rel_path
        return status

    def view(self) -> str:
        active = self._active_model()
        if active:
            return active.view()
        return ""
