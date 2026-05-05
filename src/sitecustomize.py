"""Process-level defaults for local CLI runs.

Python imports ``sitecustomize`` automatically during interpreter startup when
this ``src`` directory is on ``sys.path``. Keep this file limited to safe
environment defaults and compatibility aliases.

Use ``.env`` to configure LLM credentials — do NOT hardcode provider-specific
defaults here.
"""

from __future__ import annotations
