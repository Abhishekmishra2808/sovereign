#!/usr/bin/env python3
"""Build the Sovereign release bundle.

Assembles a single directory a user can extract and run, with no toolchain
required: three exes, the landing page, the dashboard bundle, example models and
third-party licences. Then optionally drives Inno Setup for a real installer.

Why a directory and not just exes: the UI is static files, not code. Shipping
only the exes would produce an app that starts and then serves a blank page,
which is the single most common way "it works on my machine" packaging bugs
ship.

Usage:
    python scripts/package_release.py                 # build the zip
    python scripts/package_release.py --installer     # also run Inno Setup
    python scripts/package_release.py --out D:\\dist
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Preferred binary directory per platform, most specific first.
BIN_CANDIDATES = [
    ROOT / "build-msvc" / "solver" / "Release",
    ROOT / "build-msvc" / "solver" / "Debug",
    ROOT / "build" / "solver",
    ROOT / "build" / "solver" / "Release",
]

REQUIRED_EXES = ["sovereign.exe", "sovereign-server.exe", "sovereign-launcher.exe"]

# Licences that must travel with the bundle. cpp-httplib is MIT and Monocypher is
# CC0; both require their notice to be distributed, so copying them is a legal
# obligation rather than good manners.
LICENCE_FILES = [
    ROOT / "third_party" / "cpp-httplib" / "LICENSE",
    ROOT / "third_party" / "monocypher" / "LICENSE",
]


def fail(msg: str) -> "None":
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def find_bin_dir() -> Path:
    for candidate in BIN_CANDIDATES:
        if all((candidate / exe).is_file() for exe in REQUIRED_EXES):
            return candidate
    fail(
        "could not find all three binaries.\n"
        f"  looked for: {', '.join(REQUIRED_EXES)}\n"
        f"  in: {[str(c) for c in BIN_CANDIDATES]}\n\n"
        "Build first:\n"
        '  call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"\n'
        "  cmake -S . -B build-msvc -G \"Visual Studio 17 2022\" -A x64\n"
        "  cmake --build build-msvc --config Release -j 8"
    )


def check_static() -> Path:
    static = ROOT / "api" / "static"
    required = [
        static / "index.html",  # landing page
        static / "styles.css",
        static / "main.js",
        static / "fonts" / "fonts.css",
        static / "dashboard" / "index.html",  # React dashboard
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        fail(
            "the web bundle is incomplete.\n"
            f"  missing: {', '.join(missing)}\n\n"
            "Build the front ends first:\n"
            "  python benchmarks/tools/fetch_fonts.py\n"
            "  python benchmarks/tools/fetch_site_fonts.py\n"
            "  build the separate sovereign-frontend project and copy its dist into api/static/dashboard\n"
            "  then copy site/ into api/static (see scripts/package_release.py)"
        )
    return static


def verify_offline(static: Path) -> None:
    """Refuse to ship a bundle that still reaches the network."""
    audit = ROOT / "benchmarks" / "tools" / "check_offline_assets.py"
    if not audit.is_file():
        fail("offline audit script missing")
    proc = subprocess.run(
        [sys.executable, str(audit), "--dir", str(static)], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        fail(
            "offline asset audit FAILED. The bundle still references remote origins, "
            "which breaks the core promise that Sovereign runs with the network "
            "disconnected. Fix the references and re-run."
        )
    tail = [ln for ln in proc.stdout.splitlines() if "PASS" in ln]
    print(f"  {tail[0] if tail else 'offline audit passed'}")


def assemble(stage: Path, bin_dir: Path, static: Path) -> None:
    if stage.is_symlink():
        fail("refusing to replace a symlink staging directory")
    if stage.exists():
        if stage.resolve().parent != stage.parent.resolve() or stage.name != "sovereign-release":
            fail("staging path escapes output directory")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1. Binaries
    for exe in REQUIRED_EXES:
        shutil.copy2(bin_dir / exe, stage / exe)
    print(f"  binaries : {', '.join(REQUIRED_EXES)}")

    # 2. Static UI. Copied file by file rather than with copytree+dirs_exist_ok
    # so the bundle is exactly what we intend and nothing stale survives.
    file_count = 0
    for src in static.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(static)
        dest = stage / "static" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        file_count += 1
    print(f"  ui       : {file_count} file(s) -> static/")

    # 3. Example models, so a fresh install has something to solve.
    model_count = 0
    examples = ROOT / "examples" / "models"
    if examples.is_dir():
        (stage / "examples" / "models").mkdir(parents=True, exist_ok=True)
        for src in sorted(examples.iterdir()):
            if src.suffix.lower() not in {".json", ".mps"}:
                continue
            shutil.copy2(src, stage / "examples" / "models" / src.name)
            model_count += 1
    print(f"  examples : {model_count} model(s)")

    # 4. Licences.
    lic_dir = stage / "licenses"
    lic_dir.mkdir(exist_ok=True)
    for lic in LICENCE_FILES:
        if lic.is_file():
            shutil.copy2(lic, lic_dir / lic.parent.name)
    print(f"  licenses : {len(list(lic_dir.iterdir()))} file(s)")

    json_license = ROOT / "build-msvc/_deps/nlohmann_json-src/LICENSE.MIT"
    if not json_license.is_file():
        json_license = bin_dir.parents[1] / "_deps/nlohmann_json-src/LICENSE.MIT"
    if not json_license.is_file():
        fail("nlohmann JSON license is missing from the build tree")
    shutil.copy2(json_license, lic_dir / "nlohmann-json.txt")
    import json
    frontend = ROOT.parent / "sovereign-frontend"
    lock_path = frontend / "package-lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for package, metadata in lock.get("packages", {}).items():
            if not package or metadata.get("dev"):
                continue
            directory = frontend / package
            for notice in directory.glob("*"):
                if notice.is_file() and notice.name.lower().startswith(("license", "licence", "notice")):
                    name = package.replace("node_modules/", "").replace("/", "_") + "-" + notice.name
                    shutil.copy2(notice, lic_dir / name)

    # 5. A README the user can actually read without opening the repo.
    (stage / "README.txt").write_text(
        "Sovereign - indigenous optimization engine\n"
        "=========================================\n\n"
        "QUICK START\n"
        "  Double-click  sovereign-launcher.exe\n\n"
        "  That starts a local service, opens your browser, and shows the\n"
        "  dashboard. Nothing is uploaded, nothing is phoned home, and the\n"
        "  service listens only on 127.0.0.1 - it is not reachable from your\n"
        "  network.\n\n"
        "  Closing the launcher does not stop the service. To stop it, end the\n"
        "  'sovereign-server' process in Task Manager, or run:\n"
        "      taskkill /IM sovereign-server.exe\n\n"
        "COMMAND LINE (no browser needed)\n"
        "  sovereign.exe solve examples\\models\\sample_milp.json --verify\n"
        "  sovereign.exe solve examples\\models\\sample_lp.json --verify\n"
        "  sovereign.exe version\n\n"
        "  The CLI is the engine. It works with no service running and no\n"
        "  network connection.\n\n"
        "OFFLINE\n"
        "  Everything the app needs is inside this folder: fonts, scripts,\n"
        "  styles and example models. You can disconnect the network entirely\n"
        "  and everything above still works.\n\n"
        "WHAT IT SOLVES\n"
        "  LP   - linear programming (revised simplex, Mehrotra interior point)\n"
        "  MILP - mixed integer linear (branch-and-cut, strong branching)\n"
        "  QP   - convex quadratic\n\n"
        "  MIQP, NLP and MINLP are not implemented and are not claimed.\n\n"
        "LICENCES\n"
        "  Bundled third-party notices are in licenses/.\n"
        "  The display typeface BubbledotICG-FinePos is CC BY 4.0; the landing\n"
        "  page credits its author.\n",
        encoding="utf-8",
    )
    print("  readme   : README.txt")


def make_zip(stage: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "sovereign-release.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src in sorted(stage.rglob("*")):
            if src.is_file():
                zf.write(src, src.relative_to(stage.parent))
    return zip_path


def run_installer(stage: Path, out_dir: Path) -> None:
    iscc = shutil.which("ISCC")
    if not iscc:
        for p in (
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
        ):
            if Path(p).is_file():
                iscc = p
                break
    if not iscc:
        print(
            "\n  Inno Setup (ISCC.exe) not found - skipping the installer.\n"
            "  Install it from https://jrsoftware.org/isdl.php then re-run\n"
            "  with --installer. The zip is usable as-is."
        )
        return

    script = ROOT / "installer" / "sovereign.iss"
    if not script.is_file():
        fail(f"installer script missing: {script}")
    print(f"  running  {iscc} {script.name}")
    proc = subprocess.run(
        [iscc, f"/DSourceDir={stage.parent}", f"/O{out_dir}", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        fail("Inno Setup failed")
    for produced in out_dir.glob("*.exe"):
        print(f"  installer: {produced.name}  ({produced.stat().st_size // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist"), help="output directory")
    parser.add_argument(
        "--installer", action="store_true", help="also build an installer via Inno Setup"
    )
    parser.add_argument(
        "--keep-stage", action="store_true", help="keep the staging directory"
    )
    parser.add_argument("--bin-dir", type=Path, help="explicit directory containing freshly built executables")
    args = parser.parse_args()

    print("Sovereign release packaging")
    print("-" * 60)

    print("  locating binaries...")
    bin_dir = args.bin_dir.resolve() if args.bin_dir else find_bin_dir()
    if not all((bin_dir / exe).is_file() for exe in REQUIRED_EXES):
        fail("binary directory is incomplete")
    print(f"  found    {bin_dir}")

    print("  checking web bundle...")
    static = check_static()
    print(f"  found    {static}")

    print("  verifying offline guarantee...")
    verify_offline(static)

    out_dir = Path(args.out).resolve()
    stage = out_dir / "sovereign-release"

    print("  assembling bundle...")
    assemble(stage, bin_dir, static)

    verify_offline(stage / "static")
    print("  zipping...")
    zip_path = make_zip(stage, out_dir)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  zip      {zip_path}  ({size_mb:.2f} MB)")

    if args.installer:
        print("  building installer...")
        run_installer(stage, out_dir)

    if not args.keep_stage:
        shutil.rmtree(stage, ignore_errors=True)

    print("-" * 60)
    print(f"\nRelease ready: {zip_path}")
    print("Extract it anywhere and double-click sovereign-launcher.exe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
