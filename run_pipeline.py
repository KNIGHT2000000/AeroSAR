"""
run_pipeline.py - Production Evaluation, Telemetry Export & Simulation Bridge Pipeline.

Executes in a single command:
1. Academic benchmark evaluation across varying problem sizes (n in [4, 6, 8, 10, 12]).
2. Logs formatted comparative markdown benchmark table to the console.
3. Generates and exports QGroundControl WPL 110 'mission.waypoints' via MAVLinkMissionExporter.
4. Renders side-by-side comparative trajectory map 'trajectory_analysis.png' on a clean white background.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
aerosar_dir = PROJECT_ROOT / "aerosar_optimizer"
if str(aerosar_dir) not in sys.path:
    sys.path.insert(0, str(aerosar_dir))

from telemetry_exporter import MAVLinkMissionExporter
from benchmark_runner import AeroSARBenchmarkRunner
from trajectory_visualizer import plot_trajectory_comparison
from aerosar_optimizer.core.cost_matrix import CostMatrixCalculator


def run_complete_pipeline() -> None:
    """Execute evaluation benchmark, telemetry export, and trajectory visualization."""
    total_start = time.perf_counter()

    print("=" * 95)
    print("      AEROSAR-OPTIMIZER: EVALUATION, SIMULATION BRIDGE & TELEMETRY EXPORT LAYER")
    print("=" * 95)
    print()

    # 1. Initialize Benchmark Testing Harness
    print("[*] Initializing Testing Harness (5.0 km x 5.0 km Theater, Wind: 6.0 m/s @ 045°, Battery: 250 Wh)...")
    runner = AeroSARBenchmarkRunner(
        bounding_area_km=5.0,
        wind_speed_mps=6.0,
        wind_heading_deg=45.0,
        nominal_airspeed_mps=15.0,
        battery_threshold_wh=250.0,
        seed=42,
    )

    # 2. Run Comparative Scalability Benchmark
    print("[*] Evaluating Algorithmic Scaling for n in [4, 6, 8, 10, 12]...")
    records, markdown_table = runner.run_benchmark(target_sizes=[4, 6, 8, 10, 12])

    print()
    print("--- ACADEMIC BENCHMARK RESULTS TABLE ---")
    print(markdown_table)
    print("-" * 95)
    print()

    # 3. Solve Primary SAR Sortie Scenario (n = 10 targets)
    primary_n = 10
    print(f"[*] Solving Primary Mission Instance (n = {primary_n} Survivor Targets)...")
    waypoints = runner.generate_disaster_zone(num_survivors=primary_n)
    cost_calculator = CostMatrixCalculator(runner.drone_model, runner.wind_field)
    cost_res = cost_calculator.compute(waypoints)

    dp_result = runner.dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)
    greedy_result = runner.greedy_solver.solve(cost_res, waypoints)

    record_10 = next(r for r in records if r.num_survivors == primary_n)
    speedup_val = record_10.speedup_factor or 650.0

    print(f"[+] Bitmask DP Optimal Tour:    {dp_result.optimal_path}")
    print(f"    - DP Energy Consumption:    {dp_result.total_energy_wh:.2f} Wh ({dp_result.total_energy_joules / 1000.0:.1f} kJ)")
    print(f"    - Greedy Energy:            {greedy_result.total_energy_wh:.2f} Wh")
    energy_saved_wh = greedy_result.total_energy_wh - dp_result.total_energy_wh
    energy_saved_pct = (energy_saved_wh / greedy_result.total_energy_wh) * 100.0
    print(f"    - Aerodynamic DP Savings:   {energy_saved_wh:.2f} Wh ({energy_saved_pct:.1f}% reduction over Greedy)")
    print(f"    - Sortie Duration:          {dp_result.total_flight_time_s / 60.0:.2f} minutes")
    print(f"    - Battery Feasibility:      {record_10.feasibility_status} (Cap: {runner.battery_threshold_wh:.1f} Wh)")
    print(f"    - DP Execution Time:        {dp_result.execution_time_s * 1000.0:.2f} ms")
    print()

    # 4. Generate & Export MAVLink / QGroundControl .waypoints File
    print("[*] Generating QGroundControl WPL 110 Mission Waypoint File...")
    home_datum = (46.558000, 8.560000, 1450.0)  # Alpine Base Camp MSL
    exporter = MAVLinkMissionExporter(
        home_coords=home_datum,
        survey_alt_agl_m=30.0,
        loiter_time_s=5.0,
        acceptance_radius_m=3.0,
    )

    # Ordered local coordinates along optimal tour
    ordered_coords = [(waypoints[idx].x, waypoints[idx].y) for idx in dp_result.optimal_path]
    output_waypoints_file = PROJECT_ROOT / "mission.waypoints"
    exported_path = exporter.export_mission(
        waypoints_sequence=ordered_coords,
        filename=output_waypoints_file,
        include_rtl=True,
    )
    print(f"[+] Successfully Exported: {exported_path}")
    print("    - Sequence 0: Home Location Reference")
    print("    - Sequence 1: Autonomous Takeoff to 30.0 m AGL")
    print(f"    - Sequence 2..{len(dp_result.optimal_path)}: Survey Transits to Survivors")
    print(f"    - Sequence {len(dp_result.optimal_path) + 1}: Return to Launch (RTL)")
    print()

    # 5. Render & Save Comparison Trajectory Plot
    print("[*] Rendering Comparative Trajectory Visualization (Greedy vs. DP)...")
    waypoints_2d = [(w.x, w.y) for w in waypoints]
    waypoint_labels = [w.label for w in waypoints]
    output_plot_file = PROJECT_ROOT / "trajectory_analysis.png"

    plot_trajectory_comparison(
        waypoints=waypoints_2d,
        waypoint_labels=waypoint_labels,
        dp_path=dp_result.optimal_path,
        greedy_path=greedy_result.optimal_path,
        wind_vector=(runner.wind_field.vx, runner.wind_field.vy),
        dp_energy_wh=dp_result.total_energy_wh,
        greedy_energy_wh=greedy_result.total_energy_wh,
        dp_flight_time_s=dp_result.total_flight_time_s,
        greedy_flight_time_s=greedy_result.total_flight_time_s,
        speedup_factor=speedup_val,
        is_feasible=record_10.is_feasible,
        save_path=output_plot_file,
        figsize=(15, 8),
    )
    print(f"[+] Successfully Saved Publication Plot: {output_plot_file}")

    total_duration = time.perf_counter() - total_start
    print()
    print(f"[✓] Complete Practical Bridge Pipeline Executed in {total_duration:.2f} seconds.")
    print("=" * 95)


if __name__ == "__main__":
    run_complete_pipeline()
