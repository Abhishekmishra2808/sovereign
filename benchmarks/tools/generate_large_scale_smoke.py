"""Generate a reproducible one-million-variable sparse LP smoke test.

This intentionally separates variable-count scalability from difficult
optimization structure. The model has one zero-cost lower-bounded variable per
column and no matrix rows, so a successful result tests parsing, storage,
presolve, and solution serialization at the requested variable count.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "datasets" / "ladder"


def write_model(path: Path, variables: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0

    def write(handle, text: str) -> None:
        nonlocal size
        data = text.encode("utf-8")
        handle.write(data)
        digest.update(data)
        size += len(data)

    with path.open("wb") as handle:
        write(handle, '{"problem_type":"LP","sense":"minimize","variables":[')
        for j in range(variables):
            if j:
                write(handle, ",")
            write(handle, f'{{"name":"x{j}","type":"continuous","lower_bound":0.0,"upper_bound":1e30}}')
        write(handle, '],"objective":{"linear":{}},"constraints":[]}')
    return size, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variables", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.variables <= 0:
        raise SystemExit("--variables must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lp_smoke_{args.variables // 1000}k.json"
    size, sha256 = write_model(path, args.variables)
    meta = {
        "variables": args.variables,
        "constraints": 0,
        "nonzeros": 0,
        "bytes": size,
        "sha256": sha256,
        "generator": "benchmarks/tools/generate_large_scale_smoke.py",
    }
    path.with_suffix(".meta.json").write_text(
        "{\n"
        + f'  "variables": {meta["variables"]},\n'
        + f'  "constraints": {meta["constraints"]},\n'
        + f'  "nonzeros": {meta["nonzeros"]},\n'
        + f'  "bytes": {meta["bytes"]},\n'
        + f'  "sha256": "{meta["sha256"]}",\n'
        + f'  "generator": "{meta["generator"]}"\n'
        + "}\n"
    )
    print(f"wrote {path}: vars={args.variables} bytes={size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
