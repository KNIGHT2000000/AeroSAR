"""
core/cost_matrix.py - Directed Asymmetric Energy & Kinematics Cost Matrix Calculator.

Constructs directed, strictly asymmetric cost matrices C[i][j] for UAV search-and-rescue
trajectories in aerodynamic wind fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .physics import DronePhysicsModel, Waypoint, WindField


@dataclass(slots=True)
class CostMatrixResult:
    """Holds computed directed matrices and asymmetry diagnostic metrics.

    Attributes:
        energy_matrix: N x N directed energy cost matrix in Joules (C[i][j]).
        time_matrix: N x N directed transit & loiter duration in seconds (T[i][j]).
        ground_speed_matrix: N x N effective ground speed in m/s (Vg[i][j]).
        distance_matrix: N x N Euclidean geometric distance in meters (D[i][j]).
        num_nodes: Total number of waypoints (N = n_survivors + 1).
        asymmetry_index: Dimensionless asymmetry index alpha in [0, 1].
                         0.0 indicates a symmetric metric space (zero wind),
                         > 0.0 quantifies directed aerodynamic skew.
        max_asymmetry_ratio: Peak ratio max(|C[i][j] - C[j][i]| / min(C[i][j], C[j][i])).
        is_strictly_asymmetric: True if any pair displays non-negligible asymmetry.
    """
    energy_matrix: np.ndarray
    time_matrix: np.ndarray
    ground_speed_matrix: np.ndarray
    distance_matrix: np.ndarray
    num_nodes: int
    asymmetry_index: float
    max_asymmetry_ratio: float
    is_strictly_asymmetric: bool

    def get_energy_wh(self) -> np.ndarray:
        """Return energy matrix converted to Watt-hours (Wh)."""
        return self.energy_matrix / 3600.0


class CostMatrixCalculator:
    """Computes directed kinematic cost matrices for a set of UAV SAR waypoints."""

    def __init__(self, drone_model: DronePhysicsModel, wind_field: WindField) -> None:
        self.drone_model = drone_model
        self.wind_field = wind_field

    def compute(self, waypoints: List[Waypoint]) -> CostMatrixResult:
        """Compute the full directed energy, time, speed, and distance matrices.

        Args:
            waypoints: List of Waypoints where index 0 is the Base Depot,
                       and indices 1..N-1 are survivor triage beacons.

        Returns:
            CostMatrixResult with matrices and asymmetry verification metrics.
        """
        n = len(waypoints)
        if n < 2:
            raise ValueError(f"At least 2 waypoints required to construct cost matrix (got {n}).")

        c_energy = np.zeros((n, n), dtype=np.float64)
        c_time = np.zeros((n, n), dtype=np.float64)
        c_vground = np.zeros((n, n), dtype=np.float64)
        c_dist = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    c_energy[i, j] = 0.0
                    c_time[i, j] = 0.0
                    c_vground[i, j] = self.drone_model.nominal_airspeed_mps
                    c_dist[i, j] = 0.0
                    continue

                wp_i = waypoints[i]
                wp_j = waypoints[j]

                dist = wp_i.distance_to(wp_j)
                c_dist[i, j] = dist

                e_joules, t_seconds, v_g = self.drone_model.calculate_leg_cost(
                    origin=wp_i,
                    target=wp_j,
                    wind_field=self.wind_field,
                )

                c_energy[i, j] = e_joules
                c_time[i, j] = t_seconds
                c_vground[i, j] = v_g

        # Compute Asymmetry Metrics
        asymmetry_index, max_ratio, is_asym = self._analyze_asymmetry(c_energy, n)

        return CostMatrixResult(
            energy_matrix=c_energy,
            time_matrix=c_time,
            ground_speed_matrix=c_vground,
            distance_matrix=c_dist,
            num_nodes=n,
            asymmetry_index=asymmetry_index,
            max_asymmetry_ratio=max_ratio,
            is_strictly_asymmetric=is_asym,
        )

    @staticmethod
    def _analyze_asymmetry(
        energy_matrix: np.ndarray,
        n: int,
    ) -> Tuple[float, float, bool]:
        """Compute normalized asymmetry index alpha and peak asymmetry ratio."""
        sum_abs_diff = 0.0
        sum_total = 0.0
        max_ratio = 0.0
        is_asym = False

        for i in range(n):
            for j in range(i + 1, n):
                c_ij = energy_matrix[i, j]
                c_ji = energy_matrix[j, i]

                # Ignore infinite cost pairs
                if np.isinf(c_ij) or np.isinf(c_ji):
                    continue

                diff = abs(c_ij - c_ji)
                total = c_ij + c_ji

                sum_abs_diff += diff
                sum_total += total

                min_val = min(c_ij, c_ji)
                if min_val > 1e-3:
                    ratio = diff / min_val
                    if ratio > max_ratio:
                        max_ratio = ratio

                if diff > 1e-3:
                    is_asym = True

        alpha = (sum_abs_diff / sum_total) if sum_total > 1e-6 else 0.0
        return alpha, max_ratio, is_asym
