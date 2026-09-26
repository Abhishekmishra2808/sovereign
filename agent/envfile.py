"""Load key=value pairs from a .env file into os.environ (no extra dependency)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None, override: bool = False) -> Path | None:
    if path is None:
        # Prefer CWD, then repo root (two levels up from this file: agent/ -> root)
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
    else:
        candidates = [path]

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
        return env_path
    return None
