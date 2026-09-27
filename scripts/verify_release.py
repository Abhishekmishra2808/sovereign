"""Reproduce the local release gates with bounded subprocesses.

python scripts/verify_release.py --build-dir build-session
Build C++ and the web frontend first. This command does not download dependencies.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build-session')
    args = parser.parse_args()
    build = args.build_dir.resolve()
    binary = build / 'solver/Release'
    commands = [
        ['ctest', '--test-dir', str(build), '-C', 'Release', '--output-on-failure'],
        [sys.executable, 'tests/scripts/test_offline_assets.py'],
        [sys.executable, 'tests/scripts/check_mps_corpus.py', '--exe', str(binary / 'sovereign.exe')],
        [sys.executable, 'benchmarks/tools/check_offline_assets.py'],
        [sys.executable, 'tests/scripts/smoke_release.py', '--bin-dir', str(binary)],
    ]
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True, timeout=180)
    print('All local release gates passed.')


if __name__ == '__main__':
    main()
