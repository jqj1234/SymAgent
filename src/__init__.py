"""SymAgent: A Neural-Symbolic Self-Learning Agent Framework.

For Complex Reasoning over Knowledge Graphs (SIGIR 2025).
Reference: arXiv:2502.03283v2
"""

from .evaluate import Evaluator, accuracy, f1_score, hits_at_k
from .executor import AgentExecutor, Trajectory, compute_outcome_reward
from .kg_environment import KGEnvironment
from .llm_client import LLMClient
from .planner import AgentPlanner
from .self_learning import SelfLearner, TrajectoryPool, heuristic_merge

__version__ = "1.0.0"
__all__ = [
    "KGEnvironment",
    "LLMClient",
    "AgentPlanner",
    "AgentExecutor",
    "SelfLearner",
    "TrajectoryPool",
    "Trajectory",
    "Evaluator",
    "compute_outcome_reward",
    "heuristic_merge",
    "hits_at_k",
    "f1_score",
    "accuracy",
]
