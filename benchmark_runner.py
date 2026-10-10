"""
benchmark_runner.py - Academic Algorithmic Scalability Testing Harness.

Compares exact O(n!) Brute-Force against O(n² 2ⁿ) Held-Karp Bitmask DP
across varying survivor problem sizes (n in [4, 6, 8, 10, 12]) within a
5 km x 5 km synthetic disaster search theater subject to constant wind vectors.
"""

from __future__ import annotations

import sys
import os
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
aerosar_dir = PROJECT_ROOT / "aerosar_optimizer"
if str(aerosar_dir) not in sys.path:
    sys.path.insert(0, str(aerosar_dir))

from aerosar_optimizer.core.physics import DronePhysicsModel, WindField, Waypoint
from aerosar_optimizer.core.cost_matrix import CostMatrixCalculator
from aerosar_optimizer.core.bitmask_dp import BitmaskDPSolver, OptimizationResult
from aerosar_optimizer.core.brute_force import BruteForceSolver, GreedyNearestNeighborSolver


@dataclass(slots=True)
class BenchmarkRecord:
    """Stores performance and kinematic metrics for problem size n."""
    num_survivors: int
    num_total_nodes: int
    num_permutations: int
    brute_force_time_s: Optional[float]
    dp_time_s: float
    greedy_time_s: float
    speedup_factor: Optional[float]
    optimal_distance_km: float
    dp_energy_wh: float
    dp_energy_joules: float
    is_feasible: bool
    feasibility_status: str  # "PASS" or "FAIL"
    dp_matches_brute_force: Optional[bool]
    is_bf_estimated: bool = False
    dp_path: List[int] = field(default_factory=list)
    greedy_path: List[int] = field(default_factory=list)


class AeroSARBenchmarkRunner:
    """Academic benchmark suite evaluating computational scaling and cyber-physical kinematics."""

    def __init__(
        self,
        bounding_area_km: float = 5.0,
        wind_speed_mps: float = 6.0,
        wind_heading_deg: float = 45.0,
        nominal_airspeed_mps: float = 15.0,
        battery_threshold_wh: float = 250.0,
        seed: int = 42,
    ) -> None:
        """Initialize benchmark parameters.

        Args:
            bounding_area_km: Width and height of disaster zone in km (default 5.0 km).
            wind_speed_mps: Constant atmospheric wind speed magnitude in m/s (default 6.0 m/s).
            wind_heading_deg: Constant wind heading direction in degrees (default 045°).
            nominal_airspeed_mps: Drone nominal cruise airspeed in m/s (default 15.0 m/s).
            battery_threshold_wh: Maximum usable battery capacity threshold in Wh (default 250 Wh).
            seed: RNG seed for deterministic reproduction.
        """
        self.bounding_radius_m = (bounding_area_km * 1000.0) / 2.0  # 2500 m
        self.wind_speed = wind_speed_mps
        self.wind_heading = wind_heading_deg
        self.battery_threshold_wh = battery_threshold_wh
        self.seed = seed

        # Cyber-Physical Drone Model
        self.drone_model = DronePhysicsModel(
            nominal_airspeed_mps=nominal_airspeed_mps,
            hover_power_watts=300.0,
            drag_coefficient=0.045,
            battery_capacity_wh=battery_threshold_wh / (1.0 - 0.20),  # Gross pack capacity with 20% reserve
            reserve_margin_ratio=0.20,
        )

        self.wind_field = WindField(speed_mps=wind_speed_mps, direction_deg=wind_heading_deg)
        self.dp_solver = BitmaskDPSolver(self.drone_model)
        self.greedy_solver = GreedyNearestNeighborSolver(self.drone_model)
        self.bf_solver = BruteForceSolver(self.drone_model, max_n=9)  # Run BF directly up to n=8

    def generate_disaster_zone(self, num_survivors: int) -> List[Waypoint]:
        """Generate synthetic survivor target coordinates in the 5 km x 5 km disaster zone.

        Depot (Incident Command Base) is anchored at Cartesian origin (0, 0).
        """
        rng = np.random.default_rng(self.seed + num_survivors * 7)

        # Incident Command Base at (0, 0)
        base = Waypoint.from_cartesian(
            id_=0,
            label="INCIDENT_BASE",
            x=0.0,
            y=0.0,
            datum_lat=46.5580,
            datum_lon=8.5600,
            priority=0.0,
            loiter_time_s=0.0,
            alt_agl_m=0.0,
        )

        waypoints = [base]
        for i in range(1, num_survivors + 1):
            # Uniform scatter within [-2500m, +2500m] bounding theater
            px = float(rng.uniform(-self.bounding_radius_m, self.bounding_radius_m))
            py = float(rng.uniform(-self.bounding_radius_m, self.bounding_radius_m))
            priority = float(np.round(rng.uniform(1.0, 10.0), 1))

            wp = Waypoint.from_cartesian(
                id_=i,
                label=f"SURVIVOR_{i:02d}",
                x=px,
                y=py,
                datum_lat=46.5580,
                datum_lon=8.5600,
                priority=priority,
                loiter_time_s=5.0,
                alt_agl_m=30.0,
            )
            waypoints.append(wp)

        return waypoints

    def run_benchmark(
        self,
        target_sizes: Optional[List[int]] = None,
    ) -> Tuple[List[BenchmarkRecord], str]:
        """Execute evaluation across target sizes and return records and formatted table."""
        if target_sizes is None:
            target_sizes = [4, 6, 8, 10, 12]

        records: List[BenchmarkRecord] = []
        empirical_ns_per_perm: List[float] = []

        calculator = CostMatrixCalculator(self.drone_model, self.wind_field)

        for n in target_sizes:
            waypoints = self.generate_disaster_zone(n)
            cost_res = calculator.compute(waypoints)

            # 1. Run Bitmask DP Solver
            dp_res = self.dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)

            # 2. Run Greedy Solver
            greedy_res = self.greedy_solver.solve(cost_res, waypoints)

            # Calculate total tour geometric distance (km)
            total_dist_m = 0.0
            for k in range(len(dp_res.optimal_path) - 1):
                u = dp_res.optimal_path[k]
                v = dp_res.optimal_path[k + 1]
                total_dist_m += cost_res.distance_matrix[u, v]
            dist_km = total_dist_m / 1000.0

            # 3. Brute Force Evaluation or Academic Extrapolation
            num_perms = math.factorial(n)
            bf_time: Optional[float] = None
            speedup: Optional[float] = None
            matches: Optional[bool] = None
            is_estimated = False

            if n <= 8:
                bf_res = self.bf_solver.solve(cost_res, waypoints)
                bf_time = bf_res.execution_time_s
                if bf_res.num_evaluations > 0:
                    ns_per_eval = (bf_time * 1e9) / bf_res.num_evaluations
                    empirical_ns_per_perm.append(ns_per_eval)

                matches = abs(dp_res.total_energy_joules - bf_res.total_energy_joules) < 1e-2
                speedup = (bf_time / dp_res.execution_time_s) if dp_res.execution_time_s > 0 else 1.0
            else:
                avg_ns = float(np.mean(empirical_ns_per_perm)) if empirical_ns_per_perm else 250.0
                bf_time = (num_perms * avg_ns) / 1e9
                speedup = (bf_time / dp_res.execution_time_s) if dp_res.execution_time_s > 0 else 1.0
                is_estimated = True
                matches = None

            # Feasibility validation against battery threshold
            is_pass = dp_res.total_energy_wh <= self.battery_threshold_wh
            feas_str = "PASS" if is_pass else "FAIL"

            rec = BenchmarkRecord(
                num_survivors=n,
                num_total_nodes=n + 1,
                num_permutations=num_perms,
                brute_force_time_s=bf_time,
                dp_time_s=dp_res.execution_time_s,
                greedy_time_s=greedy_res.execution_time_s,
                speedup_factor=speedup,
                optimal_distance_km=dist_km,
                dp_energy_wh=dp_res.total_energy_wh,
                dp_energy_joules=dp_res.total_energy_joules,
                is_feasible=is_pass,
                feasibility_status=feas_str,
                dp_matches_brute_force=matches,
                is_bf_estimated=is_estimated,
                dp_path=dp_res.optimal_path,
                greedy_path=greedy_res.optimal_path,
            )
            records.append(rec)

        formatted_table = self.render_academic_markdown_table(records)
        return records, formatted_table

    def render_academic_markdown_table(self, records: List[BenchmarkRecord]) -> str:
        """Format benchmark records into an academic publication markdown table."""
        lines = [
            "| Problem Size (n) | Permutations | Brute-Force Time (s) | Bitmask DP Time (s) | Speedup Factor | Optimal Dist (km) | Tour Energy (Wh) | Feasibility | Verified (Exact) |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for r in records:
            bf_str = f"{r.brute_force_time_s:.6f}" if r.brute_force_time_s is not None else "N/A"
            if r.is_bf_estimated:
                bf_str += " *"
            dp_str = f"{r.dp_time_s:.6f}"
            speedup_str = f"{r.speedup_factor:,.1f}x" if r.speedup_factor is not None else "N/A"
            verified_str = "MATCH" if r.dp_matches_brute_force is True else ("-" if r.dp_matches_brute_force is None else "FAIL")

            lines.append(
                f"| **{r.num_survivors}** | {r.num_permutations:,} | {bf_str} | {dp_str} | "
                f"**{speedup_str}** | {r.optimal_distance_km:.2f} km | {r.dp_energy_wh:.1f} Wh | "
                f"**{r.feasibility_status}** | {verified_str} |"
            )

        lines.append("")
        lines.append("*Note: Asterisk (*) indicates theoretically extrapolated O(n!) brute-force runtime based on empirical CPU baseline.*")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 95)
    print("           AEROSAR-OPTIMIZER: ACADEMIC SCALABILITY BENCHMARK SUITE")
    print("=" * 95)
    runner = AeroSARBenchmarkRunner()
    records, table = runner.run_benchmark(target_sizes=[4, 6, 8, 10, 12])
    print()
    print(table)
    print()
    print("=" * 95)
