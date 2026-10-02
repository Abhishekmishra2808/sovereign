"""Download the public benchmark corpus used by run_coverage.py.

Sources (all public, fetched unmodified):
  netlib      Full feasible Netlib LP set, uncompressed-MPS mirror
              https://github.com/coin-or-tools/Data-Netlib
              Reference optimal values: the PROBLEM SUMMARY TABLE in
              https://www.netlib.org/lp/data/readme
  miplib      Selected MIPLIB 2017 instances, https://miplib.zib.de
              Reference values: miplib2017-v37.solu (=opt= / =best= / =inf=)
  infeasible  Small instances from MIPLIB 2017 infeasible-v8.test
  qp          Maros-Meszaros convex QP set (MAT conversion of the SIF files)
              https://github.com/qpsolvers/maros_meszaros_qpbenchmark

Files land in benchmarks/datasets/coverage/ (git-ignored; regenerate with this
script). corpus.json records source URL, sha256, size and reference value for
every file so a run can be checked against exactly the same inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "datasets" / "coverage"

NETLIB_MIRROR = "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/{name}.mps.gz"
NETLIB_LIST = "https://api.github.com/repos/coin-or-tools/Data-Netlib/contents"
NETLIB_README = "https://www.netlib.org/lp/data/readme"
MIPLIB_INSTANCE = "https://miplib.zib.de/WebData/instances/{name}.mps.gz"
MIPLIB_SOLU = "https://miplib.zib.de/downloads/miplib2017-v37.solu"
MIPLIB_INFEASIBLE = "https://miplib.zib.de/downloads/infeasible-v8.test"
QP_LIST = "https://api.github.com/repos/qpsolvers/maros_meszaros_qpbenchmark/contents/data"
QP_FILE = "https://raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/main/data/{name}.mat"
# Maros & Meszaros' own 00README.QP (mirror of http://www.doc.ic.ac.uk/~im/00README.QP).
QP_README = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files/00README.QP"

# Official MIPLIB 2017 instances. Application tags only where the MIPLIB
# instance description states the application.
MIPLIB = {
    # already bundled under datasets/miplib/official
    "flugpl": "", "gt2": "", "b-ball": "", "pk1": "", "gen-ip002": "", "gen-ip016": "",
    "gen-ip021": "", "gen-ip036": "", "gen-ip054": "", "markshare_4_0": "",
    "markshare_5_0": "", "neos5": "", "mod010": "", "rentacar": "", "r50x360": "",
    "khb05250": "", "blend2": "", "binkar10_1": "", "p0201": "",
    # added for coverage
    "dcmulti": "", "fiber": "", "misc07": "", "mas74": "", "mas76": "", "noswot": "",
    "qiu": "", "qnet1": "", "qnet1_o": "", "rout": "", "10teams": "", "danoint": "",
    "neos-911970": "", "neos-3754480-nidda": "", "assign1-5-8": "", "beasleyC3": "",
    "glass4": "", "gmu-35-40": "", "pg": "", "pg5_34": "", "supportcase26": "",
    "enlight_hard": "", "neos17": "", "50v-10": "", "mik-250-20-75-4": "",
    "ran14x18-disj-8": "", "exp-1-500-5-5": "", "mcsched": "", "neos-2657525-crna": "",
    "cap6000": "", "mod011": "", "swath": "", "roll3000": "", "uct-subprob": "",
    "lotsize": "lot sizing", "tr12-30": "lot sizing",
    "p200x1188c": "fixed-charge network", "sp150x300d": "fixed-charge network",
    "eil33-2": "vehicle routing", "air03": "crew scheduling", "air04": "crew scheduling",
    "air05": "crew scheduling", "unitcal_7": "power unit commitment",
}
INFEASIBLE_MAX_BYTES = 300_000


def fetch(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sovereign-benchmarks"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha256(data)}


def netlib_reference_values() -> dict[str, dict]:
    """Parse the Optimal Value column of the Netlib PROBLEM SUMMARY TABLE."""
    text = fetch(NETLIB_README).decode("latin-1")
    start = text.index("PROBLEM SUMMARY TABLE")
    refs: dict[str, dict] = {}
    row = re.compile(r"^([A-Z0-9][A-Z0-9\-.]*)\s+\d+\s+\d+\s+\d+\s+\d+\s+(?:[BR]{1,2}\s+)?"
                     r"(-?\d\.\d+E[+-]\d+)(\s*\*\*)?")
    for line in text[start:].splitlines():
        m = row.match(line.strip())
        if m:
            value = m.group(2)
            digits = len(value.split("E")[0].replace("-", "").replace(".", ""))
            # The mirror drops dots from file names (VTP.BASE -> vtpbase).
            refs[m.group(1).lower().replace(".", "")] = {"value": float(value), "significant_digits": digits,
                                                         "approximate": bool(m.group(3))}
    return refs


def miplib_solu() -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for line in fetch(MIPLIB_SOLU).decode().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("=opt=", "=best=", "=inf=", "=unkn="):
            kind = parts[0].strip("=")
            refs[parts[1]] = {"kind": kind, "value": float(parts[2]) if len(parts) > 2 else None}
    return refs


def fetch_netlib() -> list[dict]:
    names = sorted(e["name"][:-7] for e in json.loads(fetch(NETLIB_LIST))
                   if e["name"].endswith(".mps.gz"))
    refs = netlib_reference_values()
    out_dir = OUT / "netlib"

    def one(name: str) -> dict:
        data = gzip.decompress(fetch(NETLIB_MIRROR.format(name=name)))
        entry = save(out_dir / f"{name}.mps", data)
        entry.update({"name": name, "suite": "netlib", "kind": "LP",
                      "source": NETLIB_MIRROR.format(name=name), "reference": refs.get(name)})
        return entry

    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, names))


def fetch_miplib(solu: dict) -> list[dict]:
    out_dir = OUT / "miplib"

    def one(name: str) -> dict:
        url = MIPLIB_INSTANCE.format(name=name)
        entry = save(out_dir / f"{name}.mps", gzip.decompress(fetch(url)))
        entry.update({"name": name, "suite": "miplib", "kind": "MILP", "source": url,
                      "application": MIPLIB[name] or None, "reference": solu.get(name)})
        return entry

    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, MIPLIB))


def fetch_infeasible(solu: dict) -> list[dict]:
    names = [n.strip()[:-7] for n in fetch(MIPLIB_INFEASIBLE).decode().split() if n.strip()]
    out_dir = OUT / "infeasible"

    def one(name: str) -> dict | None:
        url = MIPLIB_INSTANCE.format(name=name)
        raw = fetch(url)
        if len(raw) > INFEASIBLE_MAX_BYTES:
            return None
        entry = save(out_dir / f"{name}.mps", gzip.decompress(raw))
        entry.update({"name": name, "suite": "infeasible", "kind": "MILP", "source": url,
                      "reference": solu.get(name) or {"kind": "inf", "value": None}})
        return entry

    with cf.ThreadPoolExecutor(8) as pool:
        return [e for e in pool.map(one, names) if e]


def qp_key(name: str) -> str:
    # The readme writes cvxqp1_s as cvxqp1s.
    return name.upper().replace("_", "")


def qp_reference_values() -> dict[str, dict]:
    """OPT column of 00README.QP: BPMPD's optimum at default settings, 8 significant digits."""
    text = fetch(QP_README).decode("latin-1")
    start = text.index("NAME           M")
    row = re.compile(r"^([a-z0-9][a-z0-9\-_]*)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?\d+(?:\.\d+)?(?:e[+-]\d+)?)$",
                     re.IGNORECASE)
    refs: dict[str, dict] = {}
    for line in text[start:].splitlines():
        m = row.match(line.strip())
        if m:
            mantissa = m.group(2).lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
            refs[qp_key(m.group(1))] = {"value": float(m.group(2)), "significant_digits": len(mantissa) or 1,
                                        "approximate": False, "source": "00README.QP (BPMPD)"}
    return refs


def fetch_qp() -> list[dict]:
    names = sorted(e["name"][:-4] for e in json.loads(fetch(QP_LIST)) if e["name"].endswith(".mat"))
    refs = qp_reference_values()
    out_dir = OUT / "qp"

    def one(name: str) -> dict:
        url = QP_FILE.format(name=name)
        entry = save(out_dir / f"{name}.mat", fetch(url, timeout=300))
        entry.update({"name": name, "suite": "qp", "kind": "QP", "source": url, "reference": refs.get(qp_key(name))})
        return entry

    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="*", default=["netlib", "miplib", "infeasible", "qp"],
                    choices=["netlib", "miplib", "infeasible", "qp"])
    args = ap.parse_args()

    corpus_path = OUT / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else {}
    solu = miplib_solu() if {"miplib", "infeasible"} & set(args.suite) else {}
    fetchers = {"netlib": fetch_netlib, "miplib": lambda: fetch_miplib(solu),
                "infeasible": lambda: fetch_infeasible(solu), "qp": fetch_qp}
    for suite in args.suite:
        entries = fetchers[suite]()
        corpus[suite] = sorted(entries, key=lambda e: e["name"].lower())
        print(f"{suite}: {len(entries)} instances", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus, indent=1), encoding="utf-8")
    print(f"wrote {corpus_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
