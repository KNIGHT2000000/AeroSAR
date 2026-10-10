"""AeroSAR-Optimizer: Core Mathematical & Autonomous Robotics Modules.

Provides kinematic physics, asymmetric cost matrix computation,
exact Held-Karp bitmask dynamic programming, brute force baselines,
and MAVLink mission generation.
"""

from .physics import DronePhysicsModel, Waypoint, WindField
from .cost_matrix import CostMatrixCalculator, CostMatrixResult
from .brute_force import BruteForceSolver, GreedyNearestNeighborSolver
from .bitmask_dp import BitmaskDPSolver, OptimizationResult
from .telemetry import MAVLinkMissionGenerator

__all__ = [
    "Waypoint",
    "WindField",
    "DronePhysicsModel",
    "CostMatrixCalculator",
    "CostMatrixResult",
    "BruteForceSolver",
    "GreedyNearestNeighborSolver",
    "BitmaskDPSolver",
    "OptimizationResult",
    "MAVLinkMissionGenerator",
]
