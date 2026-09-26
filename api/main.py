from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.orchestration.agent import OptimizationAgent
from agent.schemas.models import OptimizationModel, SolverResult, VerificationResult
from agent.tools.client import ToolClient

app = FastAPI(
    title="Sovereign Optimization API",
    version="1.0.0",
    description="Tool/API interface: GPT-5.6 orchestrates; sovereign solver computes.",
)

tools = ToolClient()
agent = OptimizationAgent(tools)


class MathRequest(BaseModel):
    expression: str = Field(..., examples=["(2 + 3) * 4"])


class VerifyRequest(BaseModel):
    model: OptimizationModel
    result: SolverResult
    tol: float = 1e-6


class NLRequest(BaseModel):
    request: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "llm_enabled": agent.llm_enabled,
    }


@app.post("/tools/math_calculator")
def math_calculator(req: MathRequest) -> dict:
    try:
        return tools.math_calculator(req.expression)
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(ex)) from ex


@app.post("/solve", response_model=SolverResult)
@app.post("/tools/lp_solver", response_model=SolverResult)
@app.post("/tools/qp_solver", response_model=SolverResult)
@app.post("/tools/milp_solver", response_model=SolverResult)
def solve(model: OptimizationModel) -> SolverResult:
    if model.problem_type.value == "LP":
        return tools.lp_solver(model)
    if model.problem_type.value == "QP":
        return tools.qp_solver(model)
    return tools.milp_solver(model)


@app.post("/tools/verify_solution", response_model=VerificationResult)
def verify_solution(req: VerifyRequest) -> VerificationResult:
    return tools.verify_solution(req.model, req.result, req.tol)


@app.post("/agent/solve_structured")
def agent_solve_structured(model: OptimizationModel) -> dict:
    return agent.solve_structured(model)


@app.post("/agent/solve_nl")
def agent_solve_nl(req: NLRequest) -> dict:
    return agent.solve_natural_language(req.request)
