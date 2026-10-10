"""
core/bitmask_dp.py - Held-Karp Asymmetric Bitmask Dynamic Programming Engine.

Implements exact O(n² 2ⁿ) Held-Karp dynamic programming for the Asymmetric
Traveling Salesperson Problem (ATSP), parent backtracking for optimal trajectory
reconstruction, battery feasibility verification, and automated SAR triage logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .physics import DronePhysicsModel, Waypoint
from .cost_matrix import CostMatrixResult


@dataclass(slots=True)
class OptimizationResult:
    """Encapsulates the output of a route optimization solver.

    Attributes:
        algorithm_name: Identifier of the solver (e.g., 'Bitmask Held-Karp DP').
        optimal_path: Sequence of waypoint indices starting and ending at 0 (Depot).
        total_energy_joules: Cumulative energy expenditure for the complete closed tour.
        total_energy_wh: Total energy in Watt-hours.
        total_flight_time_s: Total mission duration including loiter times (seconds).
        execution_time_s: Solver wall-clock time in seconds.
        is_feasible: True if total energy <= usable battery capacity.
        remaining_soc_percent: Remaining battery state-of-charge upon landing at Depot.
        num_evaluations: Number of subproblem states / permutations explored.
        memory_bytes: Estimated memory allocated for the DP state table in bytes.
        leg_details: Step-by-step telemetry breakdown for each leg of the flight.
        triage_applied: True if triage pruning was triggered due to energy deficit.
        triage_notes: Diagnostic recommendations if triage was applied.
    """
    algorithm_name: str
    optimal_path: List[int]
    total_energy_joules: float
    total_energy_wh: float
    total_flight_time_s: float
    execution_time_s: float
    is_feasible: bool
    remaining_soc_percent: float
    num_evaluations: int
    memory_bytes: int = 0
    leg_details: List[Dict[str, Any]] = field(default_factory=list)
    triage_applied: bool = False
    triage_notes: Optional[str] = None


class BitmaskDPSolver:
    """Exact Held-Karp Dynamic Programming solver using bitmask state representation."""

    def __init__(self, drone_model: DronePhysicsModel) -> None:
        self.drone_model = drone_model

    def solve(
        self,
        cost_matrix_result: CostMatrixResult,
        waypoints: List[Waypoint],
        enforce_feasibility: bool = True,
    ) -> OptimizationResult:
        """Execute Held-Karp Bitmask DP to find the exact minimum-energy closed tour.

        State Formulation:
            dp[mask][u] = Minimum accumulated energy to visit all nodes in bitmask 'mask',
                          currently positioned at terminal node 'u'.
            parent[mask][u] = Predecessor node v that led to state (mask, u).

        Complexity:
            Time: O(N^2 * 2^N)
            Space: O(N * 2^N)

        Args:
            cost_matrix_result: Directed asymmetric cost matrices.
            waypoints: Full list of waypoints (Index 0 = Depot).
            enforce_feasibility: If True, triggers triage contingency if tour exceeds battery.

        Returns:
            OptimizationResult with optimal route, flight telemetry, and feasibility status.
        """
        c_matrix = cost_matrix_result.energy_matrix
        t_matrix = cost_matrix_result.time_matrix
        vg_matrix = cost_matrix_result.ground_speed_matrix
        dist_matrix = cost_matrix_result.distance_matrix

        n = len(waypoints)
        if n > 20:
            raise ValueError(
                f"Bitmask DP is restricted to N <= 20 nodes to prevent memory overflow (requested {n})."
            )

        start_time = time.perf_counter()

        num_states = 1 << n
        # DP table: 64-bit float initialized to infinity
        dp = np.full((num_states, n), np.inf, dtype=np.float64)
        # Parent backtracking table: 16-bit integer initialized to -1
        parent = np.full((num_states, n), -1, dtype=np.int16)

        memory_bytes = dp.nbytes + parent.nbytes

        # Base case: Drone starts at Depot (node 0). Mask = 1 (binary: 000...001)
        dp[1, 0] = 0.0

        eval_count = 0

        # Iterate only through odd masks (since Depot bit 0 is always visited)
        # Stepping by 2 ensures bit 0 is set: mask = 1, 3, 5, 7, ...
        for mask in range(1, num_states, 2):
            # Inspect all nodes u currently in the mask
            for u in range(n):
                if not (mask & (1 << u)):
                    continue

                cost_u = dp[mask, u]
                if np.isinf(cost_u):
                    continue

                # Transition to all unvisited nodes v
                for v in range(1, n):
                    if not (mask & (1 << v)):
                        next_mask = mask | (1 << v)
                        edge_cost = c_matrix[u, v]

                        if np.isinf(edge_cost):
                            continue

                        new_cost = cost_u + edge_cost
                        eval_count += 1

                        if new_cost < dp[next_mask, v]:
                            dp[next_mask, v] = new_cost
                            parent[next_mask, v] = u

        # Closed Tour Completion: Reconnect all candidates from full_mask back to Depot (node 0)
        full_mask = num_states - 1
        min_tour_cost = np.inf
        last_node = -1

        for u in range(1, n):
            return_cost = c_matrix[u, 0]
            if np.isinf(return_cost):
                continue

            total_candidate_cost = dp[full_mask, u] + return_cost
            if total_candidate_cost < min_tour_cost:
                min_tour_cost = total_candidate_cost
                last_node = u

        exec_time = time.perf_counter() - start_time

        # If no valid tour found (e.g. unreachable due to gale-force winds)
        if np.isinf(min_tour_cost) or last_node == -1:
            return OptimizationResult(
                algorithm_name="Bitmask Held-Karp DP",
                optimal_path=[],
                total_energy_joules=float("inf"),
                total_energy_wh=float("inf"),
                total_flight_time_s=float("inf"),
                execution_time_s=exec_time,
                is_feasible=False,
                remaining_soc_percent=0.0,
                num_evaluations=eval_count,
                memory_bytes=memory_bytes,
                triage_applied=False,
                triage_notes="Mission infeasible: no traversable path under current wind field.",
            )

        # Backtrack optimal trajectory: [0, ..., last_node, 0]
        path_rev = [0]
        curr_node = last_node
        curr_mask = full_mask

        while curr_node != 0:
            path_rev.append(curr_node)
            pred = int(parent[curr_mask, curr_node])
            if pred == -1:
                break
            # Remove current node from mask using bitwise XOR
            curr_mask = curr_mask ^ (1 << curr_node)
            curr_node = pred

        path_rev.append(0)
        optimal_path = path_rev[::-1]

        # Build telemetry and leg details
        total_time, leg_details = self._build_leg_details(
            optimal_path, waypoints, c_matrix, t_matrix, vg_matrix, dist_matrix
        )

        total_wh = min_tour_cost / 3600.0
        is_feasible = bool(self.drone_model.is_mission_feasible(min_tour_cost))
        soc_pct = float(self.drone_model.state_of_charge_percent(min_tour_cost))

        # Check for Battery Feasibility & Triage
        if not is_feasible and enforce_feasibility:
            triage_result = self._execute_triage(
                cost_matrix_result=cost_matrix_result,
                waypoints=waypoints,
                nominal_cost=min_tour_cost,
            )
            return triage_result

        return OptimizationResult(
            algorithm_name="Bitmask Held-Karp DP",
            optimal_path=optimal_path,
            total_energy_joules=float(min_tour_cost),
            total_energy_wh=float(total_wh),
            total_flight_time_s=float(total_time),
            execution_time_s=exec_time,
            is_feasible=is_feasible,
            remaining_soc_percent=float(soc_pct),
            num_evaluations=eval_count,
            memory_bytes=memory_bytes,
            leg_details=leg_details,
            triage_applied=False,
            triage_notes=None if is_feasible else (
                f"WARNING: Tour energy ({total_wh:.1f} Wh) exceeds usable envelope "
                f"({self.drone_model.usable_energy_wh:.1f} Wh by "
                f"{total_wh - self.drone_model.usable_energy_wh:.1f} Wh)."
            ),
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
        """Construct per-leg physical telemetry and cumulative metrics."""
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

    def _execute_triage(
        self,
        cost_matrix_result: CostMatrixResult,
        waypoints: List[Waypoint],
        nominal_cost: float,
    ) -> OptimizationResult:
        """Automated Triage Contingency Engine for Infeasible Battery Budgets.

        Iteratively prunes low-priority survivor targets until the optimized tour
        fits comfortably within the usable battery energy envelope.
        """
        usable_wh = self.drone_model.usable_energy_wh
        deficit_wh = (nominal_cost / 3600.0) - usable_wh

        # Sort survivor targets (indices 1..N-1) by ascending priority score
        survivor_wps = list(range(1, len(waypoints)))
        survivor_wps.sort(key=lambda idx: waypoints[idx].priority)

        current_active = set(range(len(waypoints)))
        pruned_targets: List[Waypoint] = []

        # Prune lowest priority targets one by one
        for candidate_to_prune in survivor_wps:
            current_active.remove(candidate_to_prune)
            pruned_targets.append(waypoints[candidate_to_prune])

            # Re-solve DP on active subset
            sub_waypoints = [waypoints[i] for i in sorted(current_active)]
            # Map sub-waypoint indices
            sub_index_map = {orig: new for new, orig in enumerate(sorted(current_active))}
            rev_map = {new: orig for new, orig in enumerate(sorted(current_active))}

            sub_n = len(sub_waypoints)
            sub_c = np.zeros((sub_n, sub_n), dtype=np.float64)
            sub_t = np.zeros((sub_n, sub_n), dtype=np.float64)
            sub_vg = np.zeros((sub_n, sub_n), dtype=np.float64)
            sub_d = np.zeros((sub_n, sub_n), dtype=np.float64)

            for orig_u in current_active:
                for orig_v in current_active:
                    su = sub_index_map[orig_u]
                    sv = sub_index_map[orig_v]
                    sub_c[su, sv] = cost_matrix_result.energy_matrix[orig_u, orig_v]
                    sub_t[su, sv] = cost_matrix_result.time_matrix[orig_u, orig_v]
                    sub_vg[su, sv] = cost_matrix_result.ground_speed_matrix[orig_u, orig_v]
                    sub_d[su, sv] = cost_matrix_result.distance_matrix[orig_u, orig_v]

            sub_matrix_res = CostMatrixResult(
                energy_matrix=sub_c,
                time_matrix=sub_t,
                ground_speed_matrix=sub_vg,
                distance_matrix=sub_d,
                num_nodes=sub_n,
                asymmetry_index=cost_matrix_result.asymmetry_index,
                max_asymmetry_ratio=cost_matrix_result.max_asymmetry_ratio,
                is_strictly_asymmetric=cost_matrix_result.is_strictly_asymmetric,
            )

            # Solve without recursive triage
            sub_res = self.solve(
                cost_matrix_result=sub_matrix_res,
                waypoints=sub_waypoints,
                enforce_feasibility=False,
            )

            if sub_res.is_feasible:
                # Map back to original waypoint indices
                mapped_path = [rev_map[i] for i in sub_res.optimal_path]
                pruned_names = ", ".join(f"{w.label} (Priority {w.priority:.1f})" for w in pruned_targets)
                notes = (
                    f"TRIAGE ACTIVE: Nominal mission required {nominal_cost / 3600.0:.1f} Wh "
                    f"(deficit: {deficit_wh:.1f} Wh). Automatically pruned {len(pruned_targets)} "
                    f"low-priority beacon(s) [{pruned_names}] to secure feasible SAR tour."
                )

                sub_res.algorithm_name = "Bitmask DP (Triage Pruned)"
                sub_res.optimal_path = mapped_path
                sub_res.triage_applied = True
                sub_res.triage_notes = notes
                return sub_res

        # If even pruning fails down to single beacon
        return OptimizationResult(
            algorithm_name="Bitmask DP (Triage Failed)",
            optimal_path=[],
            total_energy_joules=nominal_cost,
            total_energy_wh=nominal_cost / 3600.0,
            total_flight_time_s=0.0,
            execution_time_s=0.0,
            is_feasible=False,
            remaining_soc_percent=0.0,
            num_evaluations=0,
            triage_applied=True,
            triage_notes="CRITICAL: Even minimal single-target sorties exceed available battery budget.",
        )
