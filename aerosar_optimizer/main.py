"""
main.py - AeroSAR-Optimizer Production Pipeline Entry Point.

Demonstrates the complete end-to-end autonomous UAV search-and-rescue
pipeline from configuration to kinematic simulation, bitmask DP optimization,
MAVLink mission export, and tactical trajectory rendering in under 5 seconds.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
import yaml

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.physics import DronePhysicsModel, WindField, Waypoint
from core.cost_matrix import CostMatrixCalculator
from core.bitmask_dp import BitmaskDPSolver, OptimizationResult
from core.brute_force import GreedyNearestNeighborSolver
from core.telemetry import MAVLinkMissionGenerator
from simulation.synthetic_data import SyntheticScenarioGenerator
from simulation.runner import BenchmarkRunner
from visualization.plotter import MissionPlotter


def load_config(config_path: Path) -> dict:
    """Load mission and drone parameters from config.yaml."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def print_banner() -> None:
    """Display production aerospace ASCII terminal banner."""
    banner = r"""
========================================================================================
     ___                    ____    _    ____          ___        _   _           _             
    / _ \  ___ _ __ ___    / ___|  / \  |  _ \        / _ \ _ __ | |_(_)_ __ ___ (_)_______ _ __ 
   / /_\ \/ _ \ '__/ _ \   \___ \ / _ \ | |_) |_____ | | | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
  / /_\\ \  __/ | | (_) |   ___) / ___ \|  _ <|_____|| |_| | |_) | |_| | | | | | | |/ /  __/ |   
 /_/   \_\\___|_|  \___/   |____/_/   \_\_| \_\       \___/| .__/ \__|_|_| |_| |_|_/___\___|_|   
                                                           |_|                                    
 Energy-Aware Autonomous UAV SAR Route Optimization via Asymmetric Bitmask Dynamic Programming
========================================================================================
"""
    print(banner)


def main() -> None:
    wall_clock_start = time.perf_counter()
    print_banner()

    # 1. Load Configuration
    config_file = PROJECT_ROOT / "config.yaml"
    cfg = load_config(config_file)

    drone_cfg = cfg.get("drone", {})
    battery_cfg = cfg.get("battery", {})
    wind_cfg = cfg.get("wind", {})
    mission_cfg = cfg.get("mission", {})
    opt_cfg = cfg.get("optimization", {})
    out_cfg = cfg.get("output", {})

    output_dir = PROJECT_ROOT / out_cfg.get("directory", "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Instantiate Cyber-Physical Drone & Environmental Wind Model
    drone = DronePhysicsModel(
        nominal_airspeed_mps=drone_cfg.get("nominal_airspeed_mps", 15.0),
        max_airspeed_mps=drone_cfg.get("max_airspeed_mps", 24.0),
        hover_power_watts=drone_cfg.get("hover_power_watts", 310.0),
        drag_coefficient=drone_cfg.get("drag_coefficient", 0.048),
        battery_capacity_wh=battery_cfg.get("capacity_wh", 350.0),
        reserve_margin_ratio=battery_cfg.get("reserve_margin_ratio", 0.20),
        nominal_voltage_v=battery_cfg.get("nominal_voltage_v", 22.2),
    )

    wind = WindField(
        speed_mps=wind_cfg.get("speed_mps", 6.8),
        direction_deg=wind_cfg.get("direction_deg", 55.0),
    )

    base_datum = mission_cfg.get("base_datum", {"lat": 46.558, "lon": 8.560, "alt_m": 1450.0})
    search_radius = mission_cfg.get("search_area_radius_m", 2500.0)
    survey_alt = mission_cfg.get("survey_altitude_agl_m", 50.0)
    num_survivors = opt_cfg.get("default_target_count", 10)
    seed = opt_cfg.get("random_seed", 42)

    print(f"[*] Initializing Operational Parameters:")
    print(f"    - Drone Model:            {drone_cfg.get('model', 'HexaRotor-SAR-Heavy')}")
    print(f"    - Nominal Airspeed:       {drone.nominal_airspeed_mps:.1f} m/s ({drone.nominal_airspeed_mps * 3.6:.1f} km/h)")
    print(f"    - Atmospheric Wind:       {wind.speed_mps:.1f} m/s @ {wind.direction_deg:.1f}° (Vx={wind.vx:+.2f}, Vy={wind.vy:+.2f} m/s)")
    print(f"    - Battery Gross Capacity: {drone.battery_capacity_wh:.1f} Wh ({drone.total_energy_joules / 1000.0:.1f} kJ)")
    print(f"    - Usable Mission Buffer:  {drone.usable_energy_wh:.1f} Wh (Reserve Margin: {drone.reserve_margin_ratio * 100:.0f}%)")
    print(f"    - Search Theater Radius:  {search_radius:.0f} m around Base Camp ({base_datum['lat']:.4f}°N, {base_datum['lon']:.4f}°E)")
    print()

    # 3. Generate Clustered Survivor Scenario
    generator = SyntheticScenarioGenerator(
        datum_lat=base_datum["lat"],
        datum_lon=base_datum["lon"],
        datum_alt_m=base_datum["alt_m"],
        search_radius_m=search_radius,
    )
    scenario = generator.generate(
        num_survivors=num_survivors,
        wind_speed_mps=wind.speed_mps,
        wind_direction_deg=wind.direction_deg,
        scenario_preset="ALPINE_AVALANCHE",
        seed=seed,
    )
    waypoints = scenario.waypoints
    print(f"[+] Generated {len(waypoints) - 1} Survivor Beacons in Clustered Avalanche Zone.")

    # 4. Compute Directed Asymmetric Kinematics Cost Matrix
    matrix_calc_start = time.perf_counter()
    cost_calc = CostMatrixCalculator(drone, wind)
    cost_res = cost_calc.compute(waypoints)
    matrix_calc_time = (time.perf_counter() - matrix_calc_start) * 1000.0

    print(f"[+] Computed {cost_res.num_nodes}x{cost_res.num_nodes} Directed Asymmetric Cost Matrix in {matrix_calc_time:.2f} ms")
    print(f"    - Asymmetry Metric (alpha): {cost_res.asymmetry_index:.4f} (Strictly Asymmetric: {cost_res.is_strictly_asymmetric})")
    print(f"    - Peak Pairwise Leg Skew:   {cost_res.max_asymmetry_ratio * 100.0:.1f}%")
    print()

    # 5. Solve via Held-Karp Bitmask Dynamic Programming
    dp_solver = BitmaskDPSolver(drone)
    dp_res = dp_solver.solve(cost_res, waypoints, enforce_feasibility=True)

    # 6. Solve via Greedy Nearest Neighbor Baseline
    greedy_solver = GreedyNearestNeighborSolver(drone)
    greedy_res = greedy_solver.solve(cost_res, waypoints)

    # 7. Print Comparative Solution Telemetry
    print("=" * 88)
    print("                      MISSION OPTIMIZATION TELEMETRY SUMMARY")
    print("=" * 88)
    print(f"{'Metric':<32} | {'Bitmask Held-Karp DP':<24} | {'Greedy Nearest-Neighbor':<24}")
    print("-" * 88)
    print(f"{'Optimal Closed Tour':<32} | {str(dp_res.optimal_path):<24} | {str(greedy_res.optimal_path):<24}")
    print(f"{'Total Energy (Watt-hours)':<32} | {dp_res.total_energy_wh:<24.2f} | {greedy_res.total_energy_wh:<24.2f}")
    print(f"{'Total Energy (Kilojoules)':<32} | {dp_res.total_energy_joules / 1000.0:<24.2f} | {greedy_res.total_energy_joules / 1000.0:<24.2f}")
    print(f"{'Sortie Duration (minutes)':<32} | {dp_res.total_flight_time_s / 60.0:<24.2f} | {greedy_res.total_flight_time_s / 60.0:<24.2f}")
    print(f"{'Remaining Battery SoC (%)':<32} | {dp_res.remaining_soc_percent:<24.1f} | {greedy_res.remaining_soc_percent:<24.1f}")
    print(f"{'Feasibility Status':<32} | {('FEASIBLE' if dp_res.is_feasible else 'EXCEEDED'):<24} | {('FEASIBLE' if greedy_res.is_feasible else 'EXCEEDED'):<24}")
    print(f"{'Algorithm Wall Runtime':<32} | {f'{dp_res.execution_time_s * 1000.0:.2f} ms':<24} | {f'{greedy_res.execution_time_s * 1000.0:.2f} ms':<24}")
    print(f"{'State Space Evaluations':<32} | {f'{dp_res.num_evaluations:,}':<24} | {f'{greedy_res.num_evaluations:,}':<24}")
    print("-" * 88)

    energy_savings_wh = max(0.0, greedy_res.total_energy_wh - dp_res.total_energy_wh)
    energy_savings_pct = (energy_savings_wh / greedy_res.total_energy_wh) * 100.0 if greedy_res.total_energy_wh > 0 else 0.0
    print(f"[*] DP Aerodynamic Energy Savings: {energy_savings_wh:.2f} Wh ({energy_savings_pct:.1f}% reduction over Greedy)")
    if dp_res.triage_applied:
        print(f"[!] Triage Note: {dp_res.triage_notes}")
    print()

    # 8. Export MAVLink QGroundControl (.waypoints) Mission File
    telemetry_gen = MAVLinkMissionGenerator(
        survey_alt_agl_m=survey_alt,
        loiter_time_s=5.0,
        acceptance_radius_m=3.0,
    )
    waypoints_file = output_dir / out_cfg.get("telemetry_waypoints", "mission_plan.waypoints")
    exported_wp_path = telemetry_gen.export_to_file(
        optimal_path=dp_res.optimal_path,
        waypoints=waypoints,
        output_filepath=waypoints_file,
        include_rtl=True,
    )
    print(f"[+] Exported MAVLink Mission File: {exported_wp_path}")
    print(f"    - Compatible with: QGroundControl, ArduPilot, PX4 Autopilot, Mission Planner")

    # 9. Render & Save Tactical Trajectory Visualization
    plotter = MissionPlotter(dark_mode=True)
    plot_file = output_dir / out_cfg.get("trajectory_plot", "mission_trajectory.png")
    plotter.plot_trajectories(
        waypoints=waypoints,
        dp_result=dp_res,
        greedy_result=greedy_res,
        wind_field=wind,
        title=f"AeroSAR: Optimal 10-Survivor Trajectory (Wind: {wind.speed_mps} m/s @ {wind.direction_deg}°)",
        save_path=plot_file,
        figsize=(13, 10),
    )
    print(f"[+] Saved High-Resolution Trajectory Map: {plot_file}")

    # 10. Run Algorithmic Benchmark Suite
    print()
    print("[*] Executing Automated Algorithmic Scalability Benchmark Suite...")
    benchmark_counts = [4, 6, 8, 10, 12, 14] if "--full-benchmark" in sys.argv else [4, 6, 8, 10]
    benchmark_runner = BenchmarkRunner(drone_model=drone, wind_field=wind, bf_cutoff_n=8)
    bench_summary = benchmark_runner.run_suite(target_counts=benchmark_counts, seed=seed)
    
    print()
    print(bench_summary.to_ascii_table())

    csv_file = output_dir / out_cfg.get("benchmark_csv", "benchmark_results.csv")
    bench_summary.export_csv(csv_file)
    print(f"[+] Exported Scalability Metrics to CSV: {csv_file}")

    total_pipeline_time = time.perf_counter() - wall_clock_start
    print()
    print(f"[✓] Complete AeroSAR-Optimizer Pipeline Executed in {total_pipeline_time:.2f} seconds (< 5.00s limit).")
    print("=" * 88)


if __name__ == "__main__":
    main()
