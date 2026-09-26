"""Time OurSolver on transport LPs of growing size."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
from generate_datasets import scale_transport_lp  # noqa: E402

BIN = ROOT / "build" / "solver" / "sovereign.exe"
OUT = ROOT / "benchmarks" / "datasets" / "scale"


def main() -> None:
    sizes = [10, 20, 30, 40, 50]
    for n in sizes:
        model = scale_transport_lp(n, n)
        model.pop("notes", None)
        path = OUT / f"transport_{n}x{n}.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(BIN), "solve", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        dt = time.perf_counter() - t0
        try:
            payload = json.loads(proc.stdout)
            print(
                f"{n}x{n} vars={n*n} wall={dt:.3f}s status={payload.get('status')} "
                f"iters={payload.get('iterations')} obj={payload.get('objective_value')} "
                f"msg={str(payload.get('message',''))[:60]}"
            )
            if proc.stderr.strip():
                print(" stderr:", proc.stderr.strip().splitlines()[-3:])
        except Exception as ex:
            print(f"{n}x{n} wall={dt:.3f}s FAIL {ex} rc={proc.returncode}")
            print(proc.stderr[-500:] if proc.stderr else proc.stdout[-500:])


if __name__ == "__main__":
    main()
