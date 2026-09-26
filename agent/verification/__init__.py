"""Phase 0: verification helpers for the agent layer."""

from agent.schemas.models import OptimizationModel, SolverResult, VerificationResult
from agent.tools.client import ToolClient


def verify(model: OptimizationModel, result: SolverResult) -> VerificationResult:
    return ToolClient().verify_solution(model, result)
