"""
simulation/runner.py - Automated Benchmarking & Algorithmic Scalability Engine.

Executes comparative benchmarks across increasing survivor target counts (e.g., n in [4, 6, 8, 10, 12, 14]):
1. Measures high-precision execution runtimes (Brute Force O(n!) vs. Bitmask DP O(n² 2ⁿ) vs. Greedy O(n²)).
2. Quantifies algorithmic speedup factors and energy optimality gaps.
3. Computes memory footprint of the dynamic programming state space.
4. Renders publication-grade ASCII comparison tables and exports structured CSV data.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from core.physics import DronePhysicsModel, WindField
from core.cost_matrix import CostMatrixCalculator
from core.brute_force import BruteForceSolver, GreedyNearestNeighborSolver
from core.bitmask_dp import BitmaskDPSolver, OptimizationResult
from simulation.synthetic_data import SyntheticScenarioGenerator


@dataclass(slots=True)
class BenchmarkRow:
    """Stores comparative metrics for a specific target count n."""
    num_survivors: int
    num_total_nodes: int
    num_permutations: int
    brute_force_time_s: Optional[float]
    dp_time_s: float
    greedy_time_s: float
    speedup_factor: Optional[float]
    dp_energy_wh: float
    greedy_energy_wh: float
    energy_gap_percent: float  # (Greedy - DP) / DP * 100%
    dp_memory_kb: float
    is_feasible: bool
    dp_matches_brute_force: Optional[bool]
    is_bf_estimated: bool = False


@dataclass(slots=True)
class BenchmarkSummary:
    """Contains all benchmark evaluation rows and aggregated statistics."""
    rows: List[BenchmarkRow] = field(default_factory=list)
    measured_time_per_perm_ns: float = 250.0  # Empirical nanoseconds per permutation evaluation

    def to_ascii_table(self) -> str:
        """Render a clean, formatted ASCII table of benchmark results."""
        headers = [
            "Targets (n)",
            "States / Perms",
            "Brute Force (s)",
            "Bitmask DP (s)",
            "Speedup (x)",
            "DP Energy (Wh)",
            "Greedy Gap (%)",
            "DP Memory (KB)",
            "Feasible?",
            "Verified?",
        ]

        table_data = []
        for r in self.rows:
            bf_str = f"{r.brute_force_time_s:.6f}" if r.brute_force_time_s is not None else "N/A"
            if r.is_bf_estimated:
                bf_str += " *"

            dp_str = f"{r.dp_time_s:.6f}"
            speedup_str = f"{r.speedup_factor:,.1f}x" if r.speedup_factor is not None else "N/A"
            gap_str = f"+{r.energy_gap_percent:.1f}%"
            verified_str = "MATCH" if r.dp_matches_brute_force is True else ("FAIL" if r.dp_matches_brute_force is False else "-")
            feas_str = "YES" if r.is_feasible else "NO"

            table_data.append([
                r.num_survivors,
                f"{r.num_permutations:,}",
                bf_str,
                dp_str,
                speedup_str,
                f"{r.dp_energy_wh:.1f}",
                gap_str,
                f"{r.dp_memory_kb:,.1f}",
                feas_str,
                verified_str,
            ])

        if HAS_TABULATE:
            table_text = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        else:
            # High-quality fallback ASCII table
            col_widths = [max(len(str(val)) for val in col) for col in zip(headers, *table_data)]
            border = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
            header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
            data_lines = [
                "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) + " |"
                for row in table_data
            ]
            table_text = "\n".join([border, header_line, border] + data_lines + [border])

        footer = "\n[*] Denotes theoretically extrapolated O(n!) runtime based on empirical CPU baseline.\n"
        return table_text + footer

    def export_csv(self, filepath: str | Path) -> str:
        """Export benchmark results to CSV format."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "num_survivors",
                "total_nodes",
                "permutations_count",
                "brute_force_time_s",
                "dp_time_s",
                "greedy_time_s",
                "speedup_factor",
                "dp_energy_wh",
                "greedy_energy_wh",
                "greedy_suboptimality_gap_percent",
                "dp_memory_kb",
                "is_feasible",
                "dp_matches_brute_force",
                "is_bf_estimated",
            ])
            for r in self.rows:
                writer.writerow([
                    r.num_survivors,
                    r.num_total_nodes,
                    r.num_permutations,
                    r.brute_force_time_s if r.brute_force_time_s is not None else "",
                    r.dp_time_s,
                    r.greedy_time_s,
                    r.speedup_factor if r.speedup_factor is not None else "",
                    r.dp_energy_wh,
                    r.greedy_energy_wh,
                    r.energy_gap_percent,
                    r.dp_memory_kb,
                    r.is_feasible,
                    r.dp_matches_brute_force if r.dp_matches_brute_force is not None else "",
                    r.is_bf_estimated,
                ])

        return str(out_path.resolve())


class BenchmarkRunner:
    """Automated benchmark driver evaluating computational scaling and kinematic optimization."""

    def __init__(
        self,
        drone_model: Optional[DronePhysicsModel] = None,
        wind_field: Optional[WindField] = None,
        bf_cutoff_n: int = 8,
    ) -> None:
        self.drone_model = drone_model or DronePhysicsModel()
        self.wind_field = wind_field or WindField(speed_mps=6.8, direction_deg=55.0)
        self.bf_cutoff_n = bf_cutoff_n
        self.dp_solver = BitmaskDPSolver(self.drone_model)
        self.bf_solver = BruteForceSolver(self.drone_model, max_n=bf_cutoff_n + 1)
        self.greedy_solver = GreedyNearestNeighborSolver(self.drone_model)
        self.generator = SyntheticScenarioGenerator()

    def run_suite(
        self,
        target_counts: Optional[List[int]] = None,
        seed: int = 42,
    ) -> BenchmarkSummary:
        """Execute full comparative benchmark suite across requested target counts."""
        if target_counts is None:
            target_counts = [4, 6, 8, 10, 12, 14]

        summary = BenchmarkSummary()
        empirical_ns_per_perm: List[float] = []

        calculator = CostMatrixCalculator(self.drone_model, self.wind_field)

        for n_targets in target_counts:
            scenario = self.generator.generate(
                num_survivors=n_targets,
                wind_speed_mps=self.wind_field.speed_mps,
                wind_direction_deg=self.wind_field.direction_deg,
                scenario_preset="ALPINE_AVALANCHE",
                seed=seed,
            )

            waypoints = scenario.waypoints
            cost_matrix_res = calculator.compute(waypoints)

            # 1. Run Bitmask DP Solver
            dp_res = self.dp_solver.solve(
                cost_matrix_result=cost_matrix_res,
                waypoints=waypoints,
                enforce_feasibility=False,
            )

            # 2. Run Greedy Solver
            greedy_res = self.greedy_solver.solve(
                cost_matrix_result=cost_matrix_res,
                waypoints=waypoints,
            )

            # 3. Run or Estimate Brute Force Solver
            total_nodes = len(waypoints)
            num_perms = math.factorial(n_targets)

            bf_time: Optional[float] = None
            speedup: Optional[float] = None
            matches: Optional[bool] = None
            is_estimated = False

            if n_targets <= self.bf_cutoff_n:
                bf_res = self.bf_solver.solve(
                    cost_matrix_result=cost_matrix_res,
                    waypoints=waypoints,
                )
                bf_time = bf_res.execution_time_s
                if bf_res.num_evaluations > 0:
                    ns_per_eval = (bf_time * 1e9) / bf_res.num_evaluations
                    empirical_ns_per_perm.append(ns_per_eval)

                # Verify DP matches Brute Force exactly
                matches = abs(dp_res.total_energy_joules - bf_res.total_energy_joules) < 1e-2
                speedup = (bf_time / dp_res.execution_time_s) if dp_res.execution_time_s > 0 else 1.0
            else:
                # Estimate theoretical brute force time
                avg_ns = np.mean(empirical_ns_per_perm) if empirical_ns_per_perm else 250.0
                bf_time = (num_perms * avg_ns) / 1e9
                speedup = (bf_time / dp_res.execution_time_s) if dp_res.execution_time_s > 0 else 1.0
                is_estimated = True
                matches = None

            energy_gap_pct = (
                ((greedy_res.total_energy_joules - dp_res.total_energy_joules) / dp_res.total_energy_joules) * 100.0
                if dp_res.total_energy_joules > 0 else 0.0
            )

            dp_mem_kb = dp_res.memory_bytes / 1024.0

            row = BenchmarkRow(
                num_survivors=n_targets,
                num_total_nodes=total_nodes,
                num_permutations=num_perms,
                brute_force_time_s=bf_time,
                dp_time_s=dp_res.execution_time_s,
                greedy_time_s=greedy_res.execution_time_s,
                speedup_factor=speedup,
                dp_energy_wh=dp_res.total_energy_wh,
                greedy_energy_wh=greedy_res.total_energy_wh,
                energy_gap_percent=energy_gap_pct,
                dp_memory_kb=dp_mem_kb,
                is_feasible=dp_res.is_feasible,
                dp_matches_brute_force=matches,
                is_bf_estimated=is_estimated,
            )
            summary.rows.append(row)

        return summary


if __name__ == "__main__":
    import sys
    # Add project root to sys.path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("[*] Running AeroSAR-Optimizer Benchmarking Engine across n in [4, 6, 8, 10, 12, 14]...")
    runner = BenchmarkRunner(bf_cutoff_n=8)
    summary = runner.run_suite(target_counts=[4, 6, 8, 10, 12, 14])
    print(summary.to_ascii_table())
    out_csv = root / "output" / "benchmark_results.csv"
    summary.export_csv(out_csv)
    print(f"[+] Exported benchmark results to {out_csv}")
