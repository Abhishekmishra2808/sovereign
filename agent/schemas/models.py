from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ProblemType(str, Enum):
    LP = "LP"
    QP = "QP"
    MILP = "MILP"


class Sense(str, Enum):
    minimize = "minimize"
    maximize = "maximize"


class ConstraintSense(str, Enum):
    le = "<="
    ge = ">="
    eq = "="


class VariableType(str, Enum):
    continuous = "continuous"
    integer = "integer"
    binary = "binary"


class Variable(BaseModel):
    name: str
    type: VariableType = VariableType.continuous
    lower_bound: float = 0.0
    upper_bound: float = 1e30

    @model_validator(mode="after")
    def check_bounds(self) -> "Variable":
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"Invalid bounds for variable {self.name}")
        if self.type == VariableType.binary:
            if self.lower_bound < 0 or self.upper_bound > 1:
                raise ValueError(f"Binary variable {self.name} must be in [0, 1]")
        return self


class Objective(BaseModel):
    linear: Dict[str, float] = Field(default_factory=dict)
    quadratic: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class Constraint(BaseModel):
    name: str = ""
    linear: Dict[str, float] = Field(default_factory=dict)
    sense: ConstraintSense
    rhs: float


class OptimizationModel(BaseModel):
    problem_type: ProblemType
    sense: Sense = Sense.minimize
    variables: List[Variable]
    objective: Objective = Field(default_factory=Objective)
    constraints: List[Constraint] = Field(default_factory=list)

    @field_validator("variables")
    @classmethod
    def non_empty_variables(cls, v: List[Variable]) -> List[Variable]:
        if not v:
            raise ValueError("Model must contain at least one variable")
        return v


class SolverStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    ERROR = "ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class SolverResult(BaseModel):
    status: SolverStatus
    objective_value: Optional[float] = None
    primal: Dict[str, float] = Field(default_factory=dict)
    optimality_gap: float = 0.0
    iterations: int = 0
    nodes: int = 0
    runtime_seconds: float = 0.0
    message: str = ""
    warnings: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    is_valid: bool
    max_constraint_violation: float = 0.0
    max_bound_violation: float = 0.0
    max_integrality_violation: float = 0.0
    recomputed_objective: float = 0.0
    issues: List[str] = Field(default_factory=list)
    message: str = ""
