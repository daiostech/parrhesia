"""Entry point: python -m data_viewer"""

from pathlib import Path

from data_viewer.app import AppModel
from snaptui import Program


def main():
    project_root = Path(__file__).resolve().parent.parent
    model = AppModel(project_root)
    prog = Program(model, alt_screen=True)
    prog.run()


if __name__ == "__main__":
    main()
