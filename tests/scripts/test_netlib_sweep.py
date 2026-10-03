"""Focused tests for the benchmark-only Netlib verifier/report helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "runners"))

from run_coverage import RefModel  # noqa: E402
from run_netlib_sweep import independent_verify, expected_missing  # noqa: E402


class NetlibSweepTests(unittest.TestCase):
    def test_original_model_kkt_certificate(self) -> None:
        names = ["x", "y", "z"]
        model = RefModel(
            names=names,
            cost=np.array([1.0, 0.0, 0.0]),
            col_lo=np.array([0.0, -1e30, -1.0]),
            col_hi=np.array([2.0, 1e30, 3.0]),
            row_lo=np.array([1.0, -1e30, 0.0]),
            row_hi=np.array([1e30, 2.0, 0.0]),
            A=sp.csr_matrix(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            integer=np.zeros(3, dtype=bool),
        )
        payload = {
            "primal": {"x": 1.0, "y": 0.0, "z": 0.0},
            "dual": {"row_0": 1.0, "row_1": 0.0, "row_2": -1.0},
        }
        result = independent_verify(model, payload)
        self.assertTrue(result["pass"], result)
        self.assertLessEqual(result["primal_infeasibility"], 1e-12)
        self.assertLessEqual(result["dual_infeasibility"], 1e-12)
        self.assertLessEqual(result["complementarity"], 1e-12)

    def test_verifier_rejects_bad_row_dual(self) -> None:
        model = RefModel(
            names=["x"],
            cost=np.array([1.0]),
            col_lo=np.array([0.0]),
            col_hi=np.array([1e30]),
            row_lo=np.array([-1e30]),
            row_hi=np.array([1.0]),
            A=sp.csr_matrix([[1.0]]),
            integer=np.zeros(1, dtype=bool),
        )
        payload = {"primal": {"x": 0.0}, "dual": {"row_0": 1.0}}
        result = independent_verify(model, payload)
        self.assertFalse(result["pass"])
        self.assertGreater(result["dual_infeasibility"], 0.0)

    def test_missing_corpus_names_are_explicit(self) -> None:
        found = {
            "netlib": {"afiro": Path("afiro.mps")},
        }
        references = {
            "netlib": {"afiro": {}, "adlittle": {}},
        }
        missing = expected_missing(found, references)
        self.assertEqual(missing["netlib"], ["adlittle"])

    def test_production_solver_has_no_benchmark_solver_dependency(self) -> None:
        forbidden = ("highspy", "cvxopt", "mpmath")
        production_files = list((ROOT / "solver").rglob("*.cpp"))
        production_files += list((ROOT / "solver").rglob("*.hpp"))
        production_files += list((ROOT / "solver").rglob("CMakeLists.txt"))
        violations = []
        for path in production_files:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for token in forbidden:
                if token in text:
                    violations.append(f"{path.relative_to(ROOT)} contains {token}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
