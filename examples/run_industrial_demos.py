"""Run industrial demonstration models through the sovereign CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "industrial_refinery_lp.json",
    "industrial_blending_lp.json",
    "industrial_power_dispatch_lp.json",
    "industrial_logistics_milp.json",
    "sample_qp.json",
]


def sovereign_bin() -> str:
    for p in [ROOT / "build" / "solver" / "sovereign.exe", ROOT / "build" / "solver" / "sovereign"]:
        if p.exists():
            return str(p)
    return "sovereign"


def main() -> int:
    bin_path = sovereign_bin()
    ok = True
    for name in MODELS:
        path = ROOT / "examples" / "models" / name
        print("=" * 60, name)
        proc = subprocess.run([bin_path, "solve", str(path), "--verify"], capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            ok = False
        else:
            # crude parse last JSON-ish status
            if '"status": "ERROR"' in proc.stdout or '"status": "NOT_IMPLEMENTED"' in proc.stdout:
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
