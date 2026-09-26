"""Azure OpenAI optimization agent (orchestration only).

Numerical solves always go through ToolClient -> sovereign solver.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.schemas.models import OptimizationModel, ProblemType, Sense
from agent.tools.client import ToolClient

EventCallback = Optional[Callable[[Dict[str, Any]], None]]


SYSTEM_PROMPT = """You are the Sovereign industrial optimization agent.

CRITICAL RULES:
1. Never numerically solve the optimization problem yourself.
2. When the user gives a complete LP/MILP/QP, emit ONE JSON model in a ```json fence.
3. Do not ask for clarification if objective, variables, bounds, and constraints are already stated.
4. Bounds like x >= 0 must be set as variable lower_bound (not only as constraints).
5. upper_bound for unbounded continuous vars: use 1e30.
6. objective.linear MUST include a nonzero coefficient for every variable that appears in the objective.
7. After emitting JSON, stop. Do not explain until the solver returns.

JSON schema:
{
  "problem_type": "LP" | "QP" | "MILP",
  "sense": "maximize" | "minimize",
  "variables": [{"name":"x","type":"continuous","lower_bound":0,"upper_bound":1e30}],
  "objective": {"linear": {"x": 3.0}},
  "constraints": [{"name":"c1","linear":{"x":1},"sense":"<=","rhs":10}]
}

Example — "maximize 3x+2y s.t. x+y<=10, x,y>=0 continuous":
```json
{
  "problem_type": "LP",
  "sense": "maximize",
  "variables": [
    {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
    {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30}
  ],
  "objective": {"linear": {"x": 3, "y": 2}},
  "constraints": [
    {"name": "resource", "linear": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}
  ]
}
```
Expected optimum: x=10, y=0, objective=30.
"""


class OptimizationAgent:
    def __init__(
        self,
        tools: ToolClient | None = None,
        max_iterations: int = 6,
    ) -> None:
        self.tools = tools or ToolClient()
        self.max_iterations = max_iterations
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.endpoint and self.api_key)

    def solve_structured(
        self,
        model: OptimizationModel | Dict[str, Any],
        on_event: EventCallback = None,
    ) -> Dict[str, Any]:
        m = model if isinstance(model, OptimizationModel) else OptimizationModel.model_validate(model)
        trace: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        def emit(step: Dict[str, Any]) -> None:
            trace.append(step)
            if on_event:
                on_event(step)

        tool_name = {
            ProblemType.LP: "lp_solver",
            ProblemType.QP: "qp_solver",
            ProblemType.MILP: "milp_solver",
        }[m.problem_type]

        emit(
            {
                "kind": "tool",
                "tool": tool_name,
                "detail": f"{m.problem_type.value} | {len(m.variables)} vars",
                "status": "running",
            }
        )
        result = self._dispatch(m)
        tools_used.append(tool_name)
        emit(
            {
                "kind": "tool",
                "tool": tool_name,
                "detail": str(result.status.value),
                "status": "ok" if result.status.value != "ERROR" else "error",
            }
        )

        emit(
            {
                "kind": "tool",
                "tool": "verify_solution",
                "detail": "",
                "status": "running",
            }
        )
        verification = self.tools.verify_solution(m, result)
        tools_used.append("verify_solution")
        emit(
            {
                "kind": "tool",
                "tool": "verify_solution",
                "detail": "passed" if verification.is_valid else "failed",
                "status": "ok" if verification.is_valid else "error",
            }
        )

        return {
            "model": m.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "trace": trace,
            "tools_used": tools_used,
            "llm_enabled": self.llm_enabled,
            "pipeline": "tools_only",
        }

    def solve_natural_language(
        self,
        user_request: str,
        on_event: EventCallback = None,
    ) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        tools_used: List[str] = []

        def emit(step: Dict[str, Any]) -> None:
            trace.append(step)
            if on_event:
                on_event(step)

        if not self.llm_enabled:
            return {
                "error": "LLM not configured.",
                "hint": "Add credentials to .env, or use /load + /solve.",
                "llm_enabled": False,
                "trace": trace,
                "tools_used": tools_used,
            }

        text = user_request.strip()
        if text.startswith(">"):
            text = text.lstrip(">").strip()
        if not text:
            return {
                "answer": "Empty message — paste an optimization problem or use /help.",
                "llm_enabled": True,
                "trace": trace,
                "tools_used": tools_used,
            }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        for iteration in range(self.max_iterations):
            emit(
                {
                    "kind": "llm",
                    "tool": "formulate_model",
                    "detail": f"iteration {iteration+1}",
                    "status": "running",
                }
            )
            content = self._chat(messages)
            tools_used.append("formulate_model")
            emit(
                {
                    "kind": "llm",
                    "tool": "formulate_model",
                    "detail": "model ready" if self._extract_json(content) else "needs input",
                    "status": "ok",
                }
            )
            history.append({"iteration": iteration, "assistant": content})

            model_json = self._extract_json(content)
            if model_json is None:
                if _looks_like_question(content) or iteration == self.max_iterations - 1:
                    return {
                        "history": history,
                        "answer": content,
                        "llm_enabled": True,
                        "trace": trace,
                        "tools_used": tools_used,
                        "pipeline": "llm_clarification_only",
                    }
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Emit the OptimizationModel as a single ```json block now. "
                            "Do not ask for data that was already provided."
                        ),
                    }
                )
                continue

            try:
                model = OptimizationModel.model_validate(model_json)
            except Exception as ex:  # noqa: BLE001
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"JSON failed validation: {ex}. Emit repaired ```json only.",
                    }
                )
                continue

            quality = _model_quality_issues(model)
            if quality:
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Model quality check failed:\n- "
                            + "\n- ".join(quality)
                            + "\nEmit corrected ```json only. Keep the user's numbers."
                        ),
                    }
                )
                continue

            tool_name = {
                ProblemType.LP: "lp_solver",
                ProblemType.QP: "qp_solver",
                ProblemType.MILP: "milp_solver",
            }[model.problem_type]

            emit(
                {
                    "kind": "tool",
                    "tool": tool_name,
                    "detail": f"{model.problem_type.value} | {len(model.variables)} vars",
                    "status": "running",
                }
            )
            result = self._dispatch(model)
            tools_used.append(tool_name)
            emit(
                {
                    "kind": "tool",
                    "tool": tool_name,
                    "detail": str(result.status.value),
                    "status": "ok" if result.status.value != "ERROR" else "error",
                }
            )

            emit(
                {
                    "kind": "tool",
                    "tool": "verify_solution",
                    "detail": "",
                    "status": "running",
                }
            )
            verification = self.tools.verify_solution(model, result)
            tools_used.append("verify_solution")
            emit(
                {
                    "kind": "tool",
                    "tool": "verify_solution",
                    "detail": "passed" if verification.is_valid else "failed",
                    "status": "ok" if verification.is_valid else "error",
                }
            )

            tool_payload = {
                "model": model.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "verification": verification.model_dump(mode="json"),
            }
            history.append({"tool": tool_payload})

            suspicious = _suspicious_result(model, result)
            if suspicious:
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Solver returned a suspicious result:\n"
                            + json.dumps(tool_payload["result"])
                            + "\nIssues:\n- "
                            + "\n- ".join(suspicious)
                            + "\nYour previous JSON was likely wrong. Emit corrected ```json only."
                        ),
                    }
                )
                continue

            if result.status.value in {"OPTIMAL", "FEASIBLE"} and verification.is_valid:
                explanation = _local_explanation(model, result)
                return {
                    "model": model.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "verification": verification.model_dump(mode="json"),
                    "explanation": explanation,
                    "history": history,
                    "trace": trace,
                    "tools_used": tools_used,
                    "llm_enabled": True,
                    "pipeline": "llm_formulate -> tools_solve -> verify",
                }

            if result.status.value in {"INFEASIBLE", "UNBOUNDED"}:
                return {
                    "model": model.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "verification": verification.model_dump(mode="json"),
                    "explanation": f"Solver status: {result.status.value}. {result.message}",
                    "history": history,
                    "trace": trace,
                    "tools_used": tools_used,
                    "llm_enabled": True,
                    "pipeline": "llm_formulate -> tools_solve -> verify",
                }

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Solver output:\n"
                        + json.dumps(tool_payload["result"])
                        + "\nRepair the model JSON if appropriate."
                    ),
                }
            )

        return {
            "history": history,
            "error": "Max iterations reached",
            "llm_enabled": True,
            "trace": trace,
            "tools_used": tools_used,
        }

    def _dispatch(self, model: OptimizationModel):
        if model.problem_type == ProblemType.LP:
            return self.tools.lp_solver(model)
        if model.problem_type == ProblemType.QP:
            return self.tools.qp_solver(model)
        return self.tools.milp_solver(model)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        import urllib.error
        import urllib.request

        endpoint = self.endpoint.rstrip("/")
        if endpoint.endswith("/openai"):
            endpoint = endpoint[: -len("/openai")]

        use_v1 = os.getenv("AZURE_OPENAI_USE_V1", "").lower() in {"1", "true", "yes"}
        errors: List[str] = []

        attempts: List[Tuple[str, Dict[str, Any]]] = []
        classic = (
            f"{endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}",
            {"messages": messages},
        )
        v1 = (
            f"{endpoint}/openai/v1/chat/completions",
            {"model": self.deployment, "messages": messages},
        )
        if use_v1:
            attempts.extend([v1, classic])
        else:
            attempts.extend([classic, v1])

        for url, payload in attempts:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as ex:
                detail = ex.read().decode("utf-8", errors="replace")
                errors.append(f"{ex.code} {url}\n{detail[:400]}")
                if ex.code != 404:
                    break
            except urllib.error.URLError as ex:
                raise RuntimeError(f"LLM network error: {ex}") from ex

        raise RuntimeError(
            "LLM request failed — check .env credentials and try again."
        ) from None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass
        # Brace-balanced extract
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    return "?" in t or t.startswith("could you") or t.startswith("please provide") or t.startswith("what ")


def _model_quality_issues(model: OptimizationModel) -> List[str]:
    issues: List[str] = []
    if not model.objective.linear and not model.objective.quadratic:
        issues.append("objective is empty — include nonzero linear coefficients")
    names = {v.name for v in model.variables}
    for name, coef in model.objective.linear.items():
        if name not in names:
            issues.append(f"objective refers to unknown variable '{name}'")
        if coef == 0:
            issues.append(f"objective coefficient for '{name}' is zero")
    free = 0
    for v in model.variables:
        if v.upper_bound - v.lower_bound > 1e-12:
            free += 1
        if v.lower_bound == 0 and v.upper_bound == 0:
            issues.append(f"variable '{v.name}' is fixed at 0 (lb=ub=0) — likely wrong")
    if free == 0:
        issues.append("all variables are fixed — nothing to optimize")
    if model.problem_type == ProblemType.LP and not model.constraints and free > 0:
        # Unconstrained maximize with positive costs is unbounded — still allow minimize
        pass
    return issues


def _suspicious_result(model: OptimizationModel, result: Any) -> List[str]:
    issues: List[str] = []
    msg = (result.message or "").lower()
    if "presolve fixed all variables" in msg:
        # Only suspicious if objective claimed nonzero coeffs
        if any(abs(c) > 0 for c in model.objective.linear.values()):
            if result.objective_value == 0.0 and model.sense == Sense.maximize:
                issues.append(
                    "presolve fixed everything to objective 0 on a maximize problem — "
                    "check bounds and objective coefficients"
                )
    return issues


def _local_explanation(model: OptimizationModel, result: Any) -> str:
    parts = [f"Status: {result.status.value}."]
    if getattr(result, "objective_value", None) is not None:
        parts.append(f"Objective = {result.objective_value}.")
    if result.primal:
        assign = ", ".join(f"{k}={v}" for k, v in sorted(result.primal.items()))
        parts.append(f"Solution: {assign}.")
    parts.append("Verified by the deterministic solver/verifier (not the LLM).")
    return " ".join(parts)
