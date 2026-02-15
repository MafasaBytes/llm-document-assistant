"""
HealthDoc AI — Entry Point

Launch the medical document analysis application.
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.ui.gradio_ui import launch_gradio_app


if __name__ == "__main__":
    launch_gradio_app()
