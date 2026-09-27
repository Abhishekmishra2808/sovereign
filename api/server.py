"""HTTP bridge between the web dashboard and the Sovereign solver binary.

This module is deliberately thin and deliberately *not* the place where any
mathematics happens. It does exactly three things:

1. reports whether the compiled ``sovereign.exe`` / ``sovereign`` engine is
   present and which build flags it was compiled with;
2. runs that binary on a model file and returns its JSON verbatim;
3. serves the benchmark CSV the evidence harness already produces.

No optimization logic lives here, and no external solver is ever consulted.
Point it at the repo root and it will find the engine and the reports.

Run with:  python -m uvicorn api.server:app --port 8000
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"
MODELS = ROOT / "examples" / "models"
DATASETS = ROOT / "benchmarks" / "datasets"
STATIC = Path(__file__).resolve().parent / "static"

# Hard ceiling for any single solve launched from the browser. The dashboard is
# an interactive surface, not a batch runner; long jobs belong to the CLI.
MAX_SOLVE_SECONDS = float(os.environ.get("SOVEREIGN_API_TIMEOUT", "30"))

app = FastAPI(
    title="Sovereign Engine API",
    version="1.0.0",
    description=(
        "Thin transport layer in front of the from-scratch C++ engine. "
        "All optimization is performed by sovereign_core; nothing is proxied "
        "to an external solver."
    ),
)


# --------------------------------------------------------------------------- #
# Engine discovery
# --------------------------------------------------------------------------- #

def _engine_path() -> Path | None:
    """Locate the compiled CLI, preferring an explicit env override."""
    override = os.environ.get("SOVEREIGN_BIN")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None

    exe = "sovereign.exe" if platform.system() == "Windows" else "sovereign"
    for candidate in (
        ROOT / "build" / "solver" / exe,
        ROOT / "build" / "Release" / "solver" / exe,
        ROOT / "build" / exe,
    ):
        if candidate.is_file():
            return candidate
    return None


def _engine_build_info() -> dict[str, Any]:
    path = _engine_path()
    if path is None:
        return {
            "available": False,
            "reason": "Engine binary not found. Build it with: cmake --build build -j",
        }
    try:
        proc = subprocess.run(
            [str(path), "version"], capture_output=True, text=True, timeout=10, check=False
        )
        version = (proc.stdout or proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        version = f"unknown ({exc})"

    return {
        "available": True,
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "version": version,
        "compiler": platform.python_version() and _compiler_name(),
        "cuda": _has_cuda(),
    }


def _compiler_name() -> str:
    # The engine is C++; report the platform, not the Python build.
    system = platform.system()
    return f"{system} {platform.release()} ({platform.machine()})"


def _has_cuda() -> bool:
    """CUDA is opt-in at build time; report honestly rather than guessing."""
    return (ROOT / "build" / "CMakeCache.txt").is_file() and "SOVEREIGN_USE_CUDA=ON" in (
        ROOT / "build" / "CMakeCache.txt"
    ).read_text(errors="ignore")


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #

def _catalogue() -> list[dict[str, Any]]:
    """Every runnable model we can find, grouped by suite."""
    seen: dict[Path, dict[str, Any]] = {}

    for suite_dir, suite in ((MODELS, "examples"), (DATASETS, "benchmarks")):
        if not suite_dir.is_dir():
            continue
        for path in sorted(suite_dir.rglob("*.json")):
            try:
                model = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(model, dict) or "variables" not in model:
                continue
            rel = path.relative_to(ROOT)
            # prefer the shallowest path when the same name appears twice
            key = path.name
            if key in seen:
                continue
            seen[key] = {
                "id": path.stem,
                "label": path.stem.replace("_", " "),
                "suite": suite,
                "path": str(rel).replace("\\", "/"),
                "problemType": model.get("problem_type", "LP"),
                "rows": len(model.get("constraints", [])),
                "columns": len(model.get("variables", [])),
                "nonzeros": sum(
                    len(c.get("linear", {})) for c in model.get("constraints", [])
                ),
                "integerVars": sum(
                    1
                    for v in model.get("variables", [])
                    if v.get("type") in ("integer", "binary")
                ),
            }

    return sorted(seen.values(), key=lambda m: (m["suite"], m["id"]))


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

class SolveRequest(BaseModel):
    modelId: str = Field(..., description="Catalogue id, e.g. sample_lp or afiro")
    algorithm: str | None = Field(
        default=None,
        description="LP method override: simplex | ipm | auto (default: engine default)",
    )
    verify: bool = Field(default=True, description="Run the independent solution verifier")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": _engine_build_info(),
        "reportCount": len(_read_benchmarks()),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": _catalogue()}


@app.get("/api/benchmarks")
def benchmarks() -> dict[str, Any]:
    return {"rows": _read_benchmarks()}


@app.post("/api/solve")
def solve(req: SolveRequest) -> dict[str, Any]:
    path = _resolve_model(req.modelId)
    engine = _engine_path()
    if engine is None:
        raise HTTPException(status_code=503, detail="Solver engine binary is not built.")

    argv = [str(engine), "solve", str(path)]
    if req.verify:
        argv.append("--verify")

    env = os.environ.copy()
    env.pop("SOVEREIGN_LP_ALGORITHM", None)
    if req.algorithm in {"simplex", "ipm", "auto"}:
        env["SOVEREIGN_LP_ALGORITHM"] = req.algorithm

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=MAX_SOLVE_SECONDS, env=env, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "modelId": req.modelId,
            "status": "TIME_LIMIT",
            "message": f"Solver exceeded {MAX_SOLVE_SECONDS:.0f}s wall clock.",
            "runtimeSeconds": time.perf_counter() - started,
        }

    if proc.returncode != 0 and not proc.stdout.strip():
        raise HTTPException(
            status_code=500, detail=(proc.stderr or "solver produced no output").strip()[:2000]
        )

    payload = _parse_result(proc.stdout)
    payload["modelId"] = req.modelId
    payload["runtimeSeconds"] = time.perf_counter() - started
    payload["exitCode"] = proc.returncode
    return payload


@app.get("/api/benchmarks/latest.csv")
def benchmarks_csv() -> FileResponse:
    target = REPORTS / "latest.csv"
    if not target.is_file():
        raise HTTPException(status_code=404, detail="No benchmark report generated yet.")
    return FileResponse(target, media_type="text/csv", filename="latest.csv")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _resolve_model(model_id: str) -> Path:
    for entry in _catalogue():
        if entry["id"] == model_id:
            return ROOT / entry["path"]
    raise HTTPException(status_code=404, detail=f"Unknown model id: {model_id}")


def _parse_result(stdout: str) -> dict[str, Any]:
    """The engine prints JSON, then optionally a trailing 'verification:' block."""
    text = stdout.strip()
    verification: dict[str, Any] | None = None
    marker = "\nverification:"
    if marker in text:
        head, tail = text.split(marker, 1)
        text = head.strip()
        try:
            verification = json.loads(tail.strip())
        except json.JSONDecodeError:
            verification = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not parse solver output: {exc}"
        ) from exc
    if verification is not None:
        result["verification"] = verification
    return result


def _read_benchmarks() -> list[dict[str, Any]]:
    target = REPORTS / "latest.csv"
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "suite": raw.get("suite", ""),
                        "problem": raw.get("problem", ""),
                        "solver": raw.get("solver", ""),
                        "status": raw.get("status", ""),
                        "objective": _maybe_float(raw.get("objective")),
                        "runtimeSeconds": _maybe_float(raw.get("runtime_s")),
                        "gap": _maybe_float(raw.get("gap")),
                        "nodes": _maybe_int(raw.get("nodes")),
                        "iterations": _maybe_int(raw.get("iterations")),
                        "milpStats": raw.get("milp_stats") or None,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return rows


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Static SPA (built by `npm run build` in ../web)
# --------------------------------------------------------------------------- #

if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="spa")
