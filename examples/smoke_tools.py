"""Smoke script for Phase 0 Python tool client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.tools.client import ToolClient


def main() -> int:
    client = ToolClient()
    print("math_calculator:", client.math_calculator("(2 + 3) * 4"))

    sample = ROOT / "examples" / "models" / "sample_lp.json"
    model = json.loads(sample.read_text(encoding="utf-8"))
    result = client.lp_solver(model)
    print("lp_solver status:", result.status.value)
    print("message:", result.message)
    verification = client.verify_solution(model, result)
    print("verify_solution valid:", verification.is_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
