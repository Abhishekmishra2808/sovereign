"""Deterministic tool client for the future GPT-5.6 agent.

Phase 0: tools are real callables with strict schemas. LP/QP/MILP route to the
sovereign CLI (or in-process HTTP API). No LLM integration yet.
"""

from __future__ import annotations

import ast
import json
import operator
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

from agent.schemas.models import (
    OptimizationModel,
    SolverResult,
    SolverStatus,
    VerificationResult,
)

Number = Union[int, float]


class ToolClient:
    """Thin client around math + solver tools."""

    def __init__(
        self,
        sovereign_bin: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.sovereign_bin = sovereign_bin or self._default_sovereign_bin()
        self.api_base = api_base

    @staticmethod
    def _default_sovereign_bin() -> str:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "build" / "solver" / "sovereign.exe",
            root / "build" / "solver" / "sovereign",
            root / "build" / "sovereign.exe",
            root / "build" / "sovereign",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return "sovereign"

    def math_calculator(self, expression: str) -> Dict[str, Any]:
        """Evaluate a small arithmetic expression safely."""
        value = _safe_eval(expression)
        return {"expression": expression, "value": value}

    def lp_solver(self, model: OptimizationModel | Dict[str, Any]) -> SolverResult:
        m = _as_model(model)
        if m.problem_type.value != "LP":
            return SolverResult(
                status=SolverStatus.ERROR,
                message="lp_solver requires problem_type=LP",
            )
        return self._solve(m)

    def qp_solver(self, model: OptimizationModel | Dict[str, Any]) -> SolverResult:
        m = _as_model(model)
        if m.problem_type.value != "QP":
            return SolverResult(
                status=SolverStatus.ERROR,
                message="qp_solver requires problem_type=QP",
            )
        return self._solve(m)

    def milp_solver(self, model: OptimizationModel | Dict[str, Any]) -> SolverResult:
        m = _as_model(model)
        if m.problem_type.value != "MILP":
            return SolverResult(
                status=SolverStatus.ERROR,
                message="milp_solver requires problem_type=MILP",
            )
        return self._solve(m)

    def verify_solution(
        self,
        model: OptimizationModel | Dict[str, Any],
        result: SolverResult | Dict[str, Any],
        tol: float = 1e-6,
    ) -> VerificationResult:
        """Independent lightweight Python verifier (mirrors C++ verifier intent)."""
        m = _as_model(model)
        r = result if isinstance(result, SolverResult) else SolverResult.model_validate(result)

        if r.status in {
            SolverStatus.NOT_IMPLEMENTED,
            SolverStatus.ERROR,
            SolverStatus.INFEASIBLE,
            SolverStatus.UNBOUNDED,
        }:
            return VerificationResult(
                is_valid=False,
                issues=[f"Non-feasible solver status: {r.status.value}"],
                message="Verification skipped for non-feasible status.",
            )

        issues = []
        max_bound = 0.0
        max_cons = 0.0
        max_int = 0.0
        x = r.primal

        for v in m.variables:
            if v.name not in x:
                issues.append(f"Missing primal for {v.name}")
                continue
            val = x[v.name]
            max_bound = max(max_bound, v.lower_bound - val, val - v.upper_bound)
            if v.type.value in {"integer", "binary"}:
                max_int = max(max_int, abs(val - round(val)))

        obj = sum(coef * x.get(name, 0.0) for name, coef in m.objective.linear.items())
        for i, row in m.objective.quadratic.items():
            for j, coef in row.items():
                obj += 0.5 * coef * x.get(i, 0.0) * x.get(j, 0.0)

        for c in m.constraints:
            lhs = sum(coef * x.get(name, 0.0) for name, coef in c.linear.items())
            if c.sense.value == "<=":
                viol = lhs - c.rhs
            elif c.sense.value == ">=":
                viol = c.rhs - lhs
            else:
                viol = abs(lhs - c.rhs)
            max_cons = max(max_cons, viol)

        if max_bound > tol:
            issues.append("Bound violation exceeds tolerance.")
        if max_cons > tol:
            issues.append("Constraint violation exceeds tolerance.")
        if max_int > tol:
            issues.append("Integrality violation exceeds tolerance.")

        return VerificationResult(
            is_valid=not issues,
            max_constraint_violation=max_cons,
            max_bound_violation=max_bound,
            max_integrality_violation=max_int,
            recomputed_objective=obj,
            issues=issues,
            message="Solution verified." if not issues else "Solution failed verification.",
        )

    def _solve(self, model: OptimizationModel) -> SolverResult:
        if self.api_base:
            return self._solve_via_api(model)
        return self._solve_via_cli(model)

    def _solve_via_cli(self, model: OptimizationModel) -> SolverResult:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            model_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
            proc = subprocess.run(
                [self.sovereign_bin, "solve", str(model_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 and not proc.stdout.strip():
                return SolverResult(
                    status=SolverStatus.ERROR,
                    message=proc.stderr.strip() or f"CLI failed with code {proc.returncode}",
                )
            payload = json.loads(proc.stdout)
            return SolverResult.model_validate(payload)

    def _solve_via_api(self, model: OptimizationModel) -> SolverResult:
        import urllib.error
        import urllib.request

        data = model.model_dump_json().encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/solve",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return SolverResult.model_validate(payload)
        except urllib.error.URLError as ex:
            return SolverResult(status=SolverStatus.ERROR, message=str(ex))


def _as_model(model: OptimizationModel | Dict[str, Any]) -> OptimizationModel:
    if isinstance(model, OptimizationModel):
        return model
    return OptimizationModel.model_validate(model)


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> Number:
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsafe or unsupported expression: {expr}")

    return _eval(tree)
