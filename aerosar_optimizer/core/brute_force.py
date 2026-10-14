"""
core/brute_force.py - O(n!) Permutation Baseline & Greedy Nearest-Neighbor Solvers.

Provides:
1. Exact O(n!) BruteForceSolver enumerating all survivor permutations anchored at Depot 0.
2. Fast O(n²) GreedyNearestNeighborSolver for heuristic trajectory comparison.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .physics import DronePhysicsModel, Waypoint
from .cost_matrix import CostMatrixResult
from .bitmask_dp import OptimizationResult


class BruteForceSolver:
    """Exact O(n!) permutation baseline solver.

    Anchors Base Depot at index 0 and evaluates all (N-1)! permutations of survivor nodes.
    Used for mathematical verification of the Bitmask DP engine on small N.
    """

    def __init__(self, drone_model: DronePhysicsModel, max_n: int = 11) -> None:
        self.drone_model = drone_model
        self.max_n = max_n

    def solve(
        self,
        cost_matrix_result: CostMatrixResult,
        waypoints: List[Waypoint],
        force_run: bool = False,
    ) -> OptimizationResult:
        """Enumerate all permutations of survivor targets and identify the minimum-energy tour.

        Args:
            cost_matrix_result: Directed asymmetric cost matrices.
            waypoints: Full list of waypoints (Index 0 = Base Depot).
            force_run: If True, bypasses the max_n safety threshold.

        Returns:
            OptimizationResult containing optimal tour and execution metrics.
        """
        n = len(waypoints)
        num_survivors = n - 1

        if n > self.max_n and not force_run:
            raise ValueError(
                f"BruteForceSolver aborted: N={n} requires {math.factorial(num_survivors):,} "
                f"permutations (O(n!)), which would freeze execution. Use Bitmask DPSolver instead, "
                f"or set force_run=True."
            )

        c_matrix = cost_matrix_result.energy_matrix
        t_matrix = cost_matrix_result.time_matrix
        vg_matrix = cost_matrix_result.ground_speed_matrix
        dist_matrix = cost_matrix_result.distance_matrix

        start_time = time.perf_counter()

        min_energy = float("inf")
        best_perm: Optional[Tuple[int, ...]] = None
        eval_count = 0

        # Permute survivor indices 1..N-1
        survivor_indices = range(1, n)
        for perm in itertools.permutations(survivor_indices):
            eval_count += 1

            # Complete tour: 0 -> perm[0] -> ... -> perm[-1] -> 0
            tour_energy = c_matrix[0, perm[0]]
            is_valid = not np.isinf(tour_energy)

            if is_valid:
                for k in range(len(perm) - 1):
                    leg_cost = c_matrix[perm[k], perm[k + 1]]
                    if np.isinf(leg_cost):
                        is_valid = False
                        break
                    tour_energy += leg_cost

            if is_valid:
                return_cost = c_matrix[perm[-1], 0]
                if not np.isinf(return_cost):
                    tour_energy += return_cost
                else:
                    is_valid = False

            if is_valid and tour_energy < min_energy:
                min_energy = tour_energy
                best_perm = perm

        exec_time = time.perf_counter() - start_time

        if best_perm is None:
            return OptimizationResult(
                algorithm_name="Brute Force O(n!)",
                optimal_path=[],
                total_energy_joules=float("inf"),
                total_energy_wh=float("inf"),
                total_flight_time_s=float("inf"),
                execution_time_s=exec_time,
                is_feasible=False,
                remaining_soc_percent=0.0,
                num_evaluations=eval_count,
            )

        optimal_path = [0] + list(best_perm) + [0]

        # Compute leg details
        total_time, leg_details = self._build_leg_details(
            optimal_path, waypoints, c_matrix, t_matrix, vg_matrix, dist_matrix
        )

        total_wh = min_energy / 3600.0
        is_feasible = self.drone_model.is_mission_feasible(min_energy)
        soc_pct = self.drone_model.state_of_charge_percent(min_energy)

        return OptimizationResult(
            algorithm_name="Brute Force O(n!)",
            optimal_path=optimal_path,
            total_energy_joules=float(min_energy),
            total_energy_wh=float(total_wh),
            total_flight_time_s=float(total_time),
            execution_time_s=exec_time,
            is_feasible=is_feasible,
            remaining_soc_percent=float(soc_pct),
            num_evaluations=eval_count,
            leg_details=leg_details,
        )

    def _build_leg_details(
        self,
        path: List[int],
        waypoints: List[Waypoint],
        c_matrix: np.ndarray,
        t_matrix: np.ndarray,
        vg_matrix: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        legs = []
        cum_energy = 0.0
        cum_time = 0.0

        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]

            leg_e = float(c_matrix[u, v])
            leg_t = float(t_matrix[u, v])
            leg_vg = float(vg_matrix[u, v])
            leg_d = float(dist_matrix[u, v])

            cum_energy += leg_e
            cum_time += leg_t

            legs.append({
                "leg_index": idx + 1,
                "from_id": waypoints[u].id,
                "from_label": waypoints[u].label,
                "to_id": waypoints[v].id,
                "to_label": waypoints[v].label,
                "distance_m": leg_d,
                "ground_speed_mps": leg_vg,
                "duration_s": leg_t,
                "energy_joules": leg_e,
                "energy_wh": leg_e / 3600.0,
                "cumulative_energy_wh": cum_energy / 3600.0,
                "remaining_soc_pct": self.drone_model.state_of_charge_percent(cum_energy),
            })

        return cum_time, legs


class GreedyNearestNeighborSolver:
    """Greedy Nearest-Neighbor heuristic solver.

    Greedily transitions to the lowest energy directed unvisited node from the current waypoint,
    then returns to Base Depot. Serves as a standard comparison baseline.
    """

    def __init__(self, drone_model: DronePhysicsModel) -> None:
        self.drone_model = drone_model

    def solve(
        self,
        cost_matrix_result: CostMatrixResult,
        waypoints: List[Waypoint],
    ) -> OptimizationResult:
        """Execute greedy nearest neighbor route generation."""
        c_matrix = cost_matrix_result.energy_matrix
        t_matrix = cost_matrix_result.time_matrix
        vg_matrix = cost_matrix_result.ground_speed_matrix
        dist_matrix = cost_matrix_result.distance_matrix

        n = len(waypoints)
        start_time = time.perf_counter()

        visited = [False] * n
        path = [0]
        visited[0] = True
        curr = 0
        total_energy = 0.0
        eval_count = 0

        for _ in range(n - 1):
            best_next = -1
            best_cost = float("inf")

            for nxt in range(1, n):
                eval_count += 1
                if not visited[nxt]:
                    cost = c_matrix[curr, nxt]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = nxt

            if best_next == -1:
                break

            visited[best_next] = True
            path.append(best_next)
            total_energy += best_cost
            curr = best_next

        # Return to depot
        return_cost = c_matrix[curr, 0]
        total_energy += return_cost
        path.append(0)

        exec_time = time.perf_counter() - start_time

        # Calculate telemetry details
        legs = []
        cum_energy = 0.0
        cum_time = 0.0

        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]

            leg_e = float(c_matrix[u, v])
            leg_t = float(t_matrix[u, v])
            leg_vg = float(vg_matrix[u, v])
            leg_d = float(dist_matrix[u, v])

            cum_energy += leg_e
            cum_time += leg_t

            legs.append({
                "leg_index": idx + 1,
                "from_id": waypoints[u].id,
                "from_label": waypoints[u].label,
                "to_id": waypoints[v].id,
                "to_label": waypoints[v].label,
                "distance_m": leg_d,
                "ground_speed_mps": leg_vg,
                "duration_s": leg_t,
                "energy_joules": leg_e,
                "energy_wh": leg_e / 3600.0,
                "cumulative_energy_wh": cum_energy / 3600.0,
                "remaining_soc_pct": self.drone_model.state_of_charge_percent(cum_energy),
            })

        total_wh = total_energy / 3600.0
        is_feasible = self.drone_model.is_mission_feasible(total_energy)
        soc_pct = self.drone_model.state_of_charge_percent(total_energy)

        return OptimizationResult(
            algorithm_name="Greedy Nearest-Neighbor",
            optimal_path=path,
            total_energy_joules=float(total_energy),
            total_energy_wh=float(total_wh),
            total_flight_time_s=float(cum_time),
            execution_time_s=exec_time,
            is_feasible=is_feasible,
            remaining_soc_percent=float(soc_pct),
            num_evaluations=eval_count,
            leg_details=legs,
        )


class EuclideanNearestNeighborSolver:
    """Traditional Euclidean Distance Nearest-Neighbor heuristic solver.

    Greedily transitions to the closest unvisited node by Euclidean distance (ignoring wind drift),
    then returns to Base Depot. This simulates traditional route planners that lack aerodynamic awareness.
    """

    def __init__(self, drone_model: DronePhysicsModel) -> None:
        self.drone_model = drone_model

    def solve(
        self,
        cost_matrix_result: CostMatrixResult,
        waypoints: List[Waypoint],
    ) -> OptimizationResult:
        """Execute Euclidean distance-first greedy route generation."""
        c_matrix = cost_matrix_result.energy_matrix
        t_matrix = cost_matrix_result.time_matrix
        vg_matrix = cost_matrix_result.ground_speed_matrix
        dist_matrix = cost_matrix_result.distance_matrix

        n = len(waypoints)
        start_time = time.perf_counter()

        visited = [False] * n
        path = [0]
        visited[0] = True
        curr = 0
        total_energy = 0.0
        eval_count = 0

        for _ in range(n - 1):
            best_next = -1
            best_dist = float("inf")

            for nxt in range(1, n):
                eval_count += 1
                if not visited[nxt]:
                    d = dist_matrix[curr, nxt]
                    if d < best_dist:
                        best_dist = d
                        best_next = nxt

            if best_next == -1:
                break

            visited[best_next] = True
            path.append(best_next)
            curr = best_next

        # Return to depot
        path.append(0)
        exec_time = time.perf_counter() - start_time

        # Calculate actual physical telemetry details under wind
        legs = []
        cum_energy = 0.0
        cum_time = 0.0

        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]

            leg_e = float(c_matrix[u, v])
            leg_t = float(t_matrix[u, v])
            leg_vg = float(vg_matrix[u, v])
            leg_d = float(dist_matrix[u, v])

            cum_energy += leg_e
            cum_time += leg_t

            legs.append({
                "leg_index": idx + 1,
                "from_id": waypoints[u].id,
                "from_label": waypoints[u].label,
                "to_id": waypoints[v].id,
                "to_label": waypoints[v].label,
                "distance_m": leg_d,
                "ground_speed_mps": leg_vg,
                "duration_s": leg_t,
                "energy_joules": leg_e,
                "energy_wh": leg_e / 3600.0,
                "cumulative_energy_wh": cum_energy / 3600.0,
                "remaining_soc_pct": self.drone_model.state_of_charge_percent(cum_energy),
            })

        total_wh = cum_energy / 3600.0
        is_feasible = self.drone_model.is_mission_feasible(cum_energy)
        soc_pct = self.drone_model.state_of_charge_percent(cum_energy)

        return OptimizationResult(
            algorithm_name="Traditional Distance-First (Wind-Blind)",
            optimal_path=path,
            total_energy_joules=float(cum_energy),
            total_energy_wh=float(total_wh),
            total_flight_time_s=float(cum_time),
            execution_time_s=exec_time,
            is_feasible=is_feasible,
            remaining_soc_percent=float(soc_pct),
            num_evaluations=eval_count,
            leg_details=legs,
        )

