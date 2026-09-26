"""Minimal free-format MPS → OptimizationModel JSON converter.

Supports NAME/ROWS/COLUMNS/RHS/BOUNDS/ENDATA enough for Netlib AFIRO-class LPs
and small MIPLIB 0-1 instances. Marker rows (type N) become the objective.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOKEN = re.compile(r"\S+")


def _tokens(line: str) -> List[str]:
    # Strip MPS comments after $* in some dialects; keep simple.
    if "$*" in line:
        line = line.split("$*", 1)[0]
    return TOKEN.findall(line)


def parse_mps(text: str) -> dict:
    section = None
    name = "unnamed"
    row_type: Dict[str, str] = {}
    obj_row: Optional[str] = None
    # col -> {row: coef}
    cols: Dict[str, Dict[str, float]] = {}
    col_order: List[str] = []
    rhs: Dict[str, float] = {}
    # col -> (lb, ub); default [0, +inf]
    bounds: Dict[str, Tuple[float, float]] = {}
    integer_cols: set[str] = set()
    in_intorg = False

    def ensure_col(c: str) -> None:
        if c not in cols:
            cols[c] = {}
            col_order.append(c)
            bounds.setdefault(c, (0.0, math.inf))

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("*"):
            continue
        # Section headers are left-justified keywords
        head = line[:14].strip().upper() if len(line) >= 1 else ""
        if line[:1] not in (" ", "\t") and head in {
            "NAME",
            "ROWS",
            "COLUMNS",
            "RHS",
            "RANGES",
            "BOUNDS",
            "ENDATA",
        }:
            section = head
            if section == "NAME":
                parts = _tokens(line)
                if len(parts) >= 2:
                    name = parts[1]
            continue

        toks = _tokens(line)
        if not toks:
            continue

        if section == "ROWS":
            # type name
            if len(toks) < 2:
                continue
            rtype, rname = toks[0].upper(), toks[1]
            row_type[rname] = rtype
            if rtype == "N" and obj_row is None:
                obj_row = rname
            continue

        if section == "COLUMNS":
            # INTORG / INTEND markers (MIPLIB)
            if len(toks) >= 3 and toks[1].upper() == "'MARKER'":
                marker = toks[2].upper().strip("'")
                if "INTORG" in marker:
                    in_intorg = True
                elif "INTEND" in marker:
                    in_intorg = False
                continue
            # col row val [row val ...]
            col = toks[0]
            ensure_col(col)
            if in_intorg:
                integer_cols.add(col)
            rest = toks[1:]
            i = 0
            while i + 1 < len(rest):
                rname, val_s = rest[i], rest[i + 1]
                cols[col][rname] = cols[col].get(rname, 0.0) + float(val_s)
                i += 2
            continue

        if section == "RHS":
            # rhsname row val [row val ...] — skip rhsname
            rest = toks[1:] if len(toks) >= 3 else toks
            # If first token looks like a row that exists, include it
            if toks and toks[0] in row_type:
                rest = toks
            elif len(toks) >= 3:
                rest = toks[1:]
            i = 0
            while i + 1 < len(rest):
                rname, val_s = rest[i], rest[i + 1]
                if rname in row_type:
                    rhs[rname] = float(val_s)
                i += 2
            continue

        if section == "BOUNDS":
            # type boundname col [value]
            if len(toks) < 3:
                continue
            btype = toks[0].upper()
            col = toks[2] if len(toks) >= 3 else toks[1]
            ensure_col(col)
            lb, ub = bounds.get(col, (0.0, math.inf))
            val = float(toks[3]) if len(toks) >= 4 else None
            if btype == "LO":
                lb = float(val) if val is not None else lb
            elif btype == "UP":
                ub = float(val) if val is not None else ub
            elif btype == "FX":
                lb = ub = float(val) if val is not None else lb
            elif btype == "FR":
                lb, ub = -math.inf, math.inf
            elif btype == "MI":
                lb = -math.inf
            elif btype == "PL":
                ub = math.inf
            elif btype == "BV":
                lb, ub = 0.0, 1.0
                integer_cols.add(col)
            elif btype == "LI":
                lb = float(val) if val is not None else 0.0
                integer_cols.add(col)
            elif btype == "UI":
                ub = float(val) if val is not None else ub
                integer_cols.add(col)
            bounds[col] = (lb, ub)
            continue

    if obj_row is None:
        raise ValueError("MPS has no objective (N) row")

    # Build objective
    objective: Dict[str, float] = {}
    for c, coeffs in cols.items():
        if obj_row in coeffs and abs(coeffs[obj_row]) > 0:
            objective[c] = coeffs[obj_row]

    # Variables
    BIG = 1e30
    variables = []
    has_integer = bool(integer_cols)
    for c in col_order:
        lb, ub = bounds.get(c, (0.0, math.inf))
        if math.isinf(lb) and lb < 0:
            lb_out = -BIG
        else:
            lb_out = 0.0 if math.isinf(lb) and lb > 0 else lb
        if math.isinf(ub):
            ub_out = BIG
        else:
            ub_out = ub
        if c in integer_cols and lb_out == 0 and ub_out == 1:
            vtype = "binary"
        elif c in integer_cols:
            vtype = "integer"
        else:
            vtype = "continuous"
        variables.append(
            {
                "name": c,
                "type": vtype,
                "lower_bound": lb_out,
                "upper_bound": ub_out,
            }
        )

    # Constraints (skip objective row)
    constraints = []
    for rname, rtype in row_type.items():
        if rtype == "N":
            continue
        linear: Dict[str, float] = {}
        for c, coeffs in cols.items():
            if rname in coeffs and abs(coeffs[rname]) > 0:
                linear[c] = coeffs[rname]
        if not linear:
            continue
        sense = {"E": "=", "L": "<=", "G": ">="}.get(rtype)
        if sense is None:
            continue
        constraints.append(
            {
                "name": rname,
                "linear": linear,
                "sense": sense,
                "rhs": float(rhs.get(rname, 0.0)),
            }
        )

    return {
        "name": name,
        "problem_type": "MILP" if has_integer else "LP",
        "sense": "minimize",
        "variables": variables,
        "objective": {"linear": objective},
        "constraints": constraints,
        "metadata": {
            "source_format": "MPS",
            "objective_row": obj_row,
            "n_vars": len(variables),
            "n_cons": len(constraints),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert MPS to sovereign JSON")
    ap.add_argument("mps", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    model = parse_mps(args.mps.read_text(encoding="utf-8", errors="replace"))
    # Strip converter-only metadata for solver (keep name in file via sidecar note)
    meta = model.pop("metadata", None)
    name = model.pop("name", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "name": name, "meta": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
