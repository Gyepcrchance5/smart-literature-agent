#!/usr/bin/env python3
"""兼容入口：把旧的 ``python start.py`` 转发到无菜单 CLI。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


if __name__ == "__main__":
    from run import main as pipeline_main

    pipeline_main()
