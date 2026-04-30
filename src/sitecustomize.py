"""Process-level defaults for local CLI runs.

Python imports ``sitecustomize`` automatically during interpreter startup when
this ``src`` directory is on ``sys.path``. Keep this file limited to safe
environment defaults and compatibility aliases.
"""

from __future__ import annotations

import os


DEFAULT_MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"


os.environ.setdefault("LLM_MODEL", "MiniMax-M2.7")
os.environ.setdefault("ANTHROPIC_BASE_URL", DEFAULT_MINIMAX_ANTHROPIC_BASE_URL)
