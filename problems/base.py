"""Base abstractions for the LLM conjecture harness.

A Problem defines two roles:
  - Strategy: an approach to solving/proving something (e.g., an estimator)
  - Instance: a test case or potential counterexample (e.g., an oracle algorithm)

The harness evaluates every strategy on every instance and displays results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating one strategy on one instance."""
    metrics: dict[str, float]        # e.g. {"cumulative_risk": 0.95, "max_Ed": 0.12}
    summary: str                     # one-line human-readable summary
    details: dict[str, Any] = field(default_factory=dict)  # anything extra (per-stage data, etc.)
    conjecture_holds: bool | None = None  # True/False/None if inconclusive


@dataclass
class Entry:
    """A strategy or instance submitted to the harness."""
    id: str
    role: str               # "strategy" or "instance"
    problem_id: str
    name: str
    description: str         # natural language description
    code: str                # generated Python code
    created_at: str
    tunable_params: dict = field(default_factory=dict)  # {name: {default, min, max, type, description}}
    param_values: dict = field(default_factory=dict)     # current overrides {name: value}


class Problem(ABC):
    """Abstract base for a conjecture/problem."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Concise problem statement for display."""

    @abstractmethod
    def strategy_spec(self) -> str:
        """Full prompt context telling the LLM how to write strategy code.
        Should include the class/function interface, constraints, and examples."""

    @abstractmethod
    def instance_spec(self) -> str:
        """Full prompt context telling the LLM how to write instance code."""

    @abstractmethod
    def evaluate(self, strategy_code: str, instance_code: str, params: dict | None = None,
                 strategy_params: dict | None = None, instance_params: dict | None = None) -> EvalResult:
        """Run the strategy on the instance and return metrics."""

    @abstractmethod
    def default_params(self) -> dict:
        """Default evaluation parameters (e.g. N, num_samples)."""
