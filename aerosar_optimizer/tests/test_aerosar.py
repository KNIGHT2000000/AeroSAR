"""
tests/test_aerosar.py - Comprehensive Unit & Integration Test Suite.

Verifies:
1. Kinematic wind triangle drift, ground speed, and aerodynamic power equations.
2. Directed asymmetric cost matrix generation and asymmetry diagnostics.
3. Held-Karp Bitmask DP exactness and equivalence to O(n!) Brute Force baseline.
4. Battery feasibility thresholds and automated triage contingency pruning.
5. QGroundControl / MAVLink WPL 110 telemetry generation and syntactic compliance.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.physics import Waypoint, WindField, DronePhysicsModel
from core.cost_matrix import CostMatrixCalculator
from core.brute_force import BruteForceSolver, GreedyNearestNeighborSolver
from core.bitmask_dp import BitmaskDPSolver
from core.telemetry import MAVLinkMissionGenerator
from simulation.synthetic_data import SyntheticScenarioGenerator


@pytest.fixture
def drone() -> DronePhysicsModel:
    return DronePhysicsModel(
        nominal_airspeed_mps=15.0,
        hover_power_watts=300.0,
        drag_coefficient=0.045,
        battery_capacity_wh=320.0,
        reserve_margin_ratio=0.20,
    )


@pytest.fixture
def wind() -> WindField:
    # 6 m/s wind towards North-East (45 deg)
    return WindField(speed_mps=6.0, direction_deg=45.0)


# ==============================================================================
# PHASE 1 TESTS: Aerodynamic Physics & Cost Matrix
# ==============================================================================

def test_wind_triangle_tailwind_vs_headwind(drone: DronePhysicsModel) -> None:
    """Verify tailwind increases ground speed and headwind decreases ground speed."""
    # Wind vector blowing East (+X) at 5 m/s
    wind_east = WindField(speed_mps=5.0, direction_deg=0.0)

    # Flying East (+X): Pure Tailwind
    disp_east = np.array([1000.0, 0.0])
    vg_tailwind, _, _ = drone.solve_wind_triangle(disp_east, wind_east)

    # Flying West (-X): Pure Headwind
    disp_west = np.array([-1000.0, 0.0])
    vg_headwind, _, _ = drone.solve_wind_triangle(disp_west, wind_east)

    assert vg_tailwind > drone.nominal_airspeed_mps
    assert vg_headwind < drone.nominal_airspeed_mps
    assert math.isclose(vg_tailwind, drone.nominal_airspeed_mps + 5.0, rel_tol=1e-3)
    assert math.isclose(vg_headwind, drone.nominal_airspeed_mps - 5.0, rel_tol=1e-3)


def test_extreme_crosswind_infeasibility(drone: DronePhysicsModel) -> None:
    """Verify exception is raised when crosswind exceeds maximum airspeed."""
    # Hurricane crosswind (30 m/s > nominal 15 m/s)
    hurricane_wind = WindField(speed_mps=30.0, direction_deg=90.0)
    disp_east = np.array([500.0, 0.0])

    with pytest.raises(ValueError, match="Aerodynamic stall/infeasibility"):
        drone.solve_wind_triangle(disp_east, hurricane_wind)


def test_power_model_cubic_drag(drone: DronePhysicsModel) -> None:
    """Verify non-linear electrical power consumption equation."""
    # P = P_hover + c_drag * V_air^3
    v = 15.0
    expected_p = 300.0 + 0.045 * (15.0 ** 3)
    assert math.isclose(drone.power_consumption(v), expected_p, rel_tol=1e-5)


def test_cost_matrix_asymmetry_with_wind(drone: DronePhysicsModel, wind: WindField) -> None:
    """Verify cost matrix is strictly asymmetric in the presence of wind."""
    wp0 = Waypoint.from_cartesian(0, "DEPOT", 0.0, 0.0, 46.5, 8.5)
    wp1 = Waypoint.from_cartesian(1, "WP1", 800.0, 800.0, 46.5, 8.5)
    wp2 = Waypoint.from_cartesian(2, "WP2", -500.0, 700.0, 46.5, 8.5)

    calculator = CostMatrixCalculator(drone, wind)
    res = calculator.compute([wp0, wp1, wp2])

    assert res.is_strictly_asymmetric
    assert res.asymmetry_index > 0.01

    # Specifically check C[0][1] != C[1][0]
    assert not math.isclose(res.energy_matrix[0, 1], res.energy_matrix[1, 0], rel_tol=1e-2)


def test_cost_matrix_symmetry_zero_wind(drone: DronePhysicsModel) -> None:
    """Verify cost matrix is symmetric when wind is zero and loiter times equal."""
    zero_wind = WindField(speed_mps=0.0, direction_deg=0.0)
    wp0 = Waypoint.from_cartesian(0, "DEPOT", 0.0, 0.0, 46.5, 8.5, loiter_time_s=5.0)
    wp1 = Waypoint.from_cartesian(1, "WP1", 500.0, 0.0, 46.5, 8.5, loiter_time_s=5.0)

    calculator = CostMatrixCalculator(drone, zero_wind)
    res = calculator.compute([wp0, wp1])

    assert math.isclose(res.energy_matrix[0, 1], res.energy_matrix[1, 0], rel_tol=1e-4)


# ==============================================================================
# PHASE 2 TESTS: Algorithmic Verification (DP vs Brute Force)
# ==============================================================================

def test_bitmask_dp_matches_brute_force_small_n(drone: DronePhysicsModel, wind: WindField) -> None:
    """Verify Bitmask DP produces the exact same minimum energy as Brute Force."""
    gen = SyntheticScenarioGenerator()
    # 5 targets (total N = 6 nodes)
    scenario = gen.generate(
        num_survivors=5,
        wind_speed_mps=wind.speed_mps,
        wind_direction_deg=wind.direction_deg,
        seed=101,
    )
    waypoints = scenario.waypoints

    calculator = CostMatrixCalculator(drone, wind)
    cost_res = calculator.compute(waypoints)

    bf_solver = BruteForceSolver(drone)
    dp_solver = BitmaskDPSolver(drone)

    bf_res = bf_solver.solve(cost_res, waypoints)
    dp_res = dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)

    # Verify identical optimal cost
    assert math.isclose(bf_res.total_energy_joules, dp_res.total_energy_joules, rel_tol=1e-4)
    assert math.isclose(bf_res.total_energy_wh, dp_res.total_energy_wh, rel_tol=1e-4)

    # Verify tour valid structure: starts and ends at Depot 0, visits all nodes
    assert dp_res.optimal_path[0] == 0
    assert dp_res.optimal_path[-1] == 0
    assert len(dp_res.optimal_path) == len(waypoints) + 1
    assert set(dp_res.optimal_path) == set(range(len(waypoints)))


def test_greedy_is_suboptimal_or_equal(drone: DronePhysicsModel, wind: WindField) -> None:
    """Verify Greedy solver never beats the exact DP optimal solver."""
    gen = SyntheticScenarioGenerator()
    scenario = gen.generate(num_survivors=8, wind_speed_mps=wind.speed_mps, wind_direction_deg=wind.direction_deg, seed=42)
    waypoints = scenario.waypoints

    calculator = CostMatrixCalculator(drone, wind)
    cost_res = calculator.compute(waypoints)

    dp_solver = BitmaskDPSolver(drone)
    greedy_solver = GreedyNearestNeighborSolver(drone)

    dp_res = dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)
    greedy_res = greedy_solver.solve(cost_res, waypoints)

    assert dp_res.total_energy_joules <= greedy_res.total_energy_joules + 1e-4


# ==============================================================================
# PHASE 3 TESTS: Battery Feasibility & Automated Triage
# ==============================================================================

def test_battery_feasibility_validation(wind: WindField) -> None:
    """Verify mission feasibility flag reflects battery capacity."""
    gen = SyntheticScenarioGenerator()
    scenario = gen.generate(num_survivors=6, wind_speed_mps=wind.speed_mps, wind_direction_deg=wind.direction_deg, seed=42)
    waypoints = scenario.waypoints

    # Generous battery -> Feasible
    big_drone = DronePhysicsModel(battery_capacity_wh=1000.0)
    calc = CostMatrixCalculator(big_drone, wind)
    cost_res = calc.compute(waypoints)

    dp_solver = BitmaskDPSolver(big_drone)
    res_feasible = dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)
    assert res_feasible.is_feasible is True

    # Tiny battery -> Infeasible
    tiny_drone = DronePhysicsModel(battery_capacity_wh=10.0)
    dp_solver_tiny = BitmaskDPSolver(tiny_drone)
    res_infeasible = dp_solver_tiny.solve(cost_res, waypoints, enforce_feasibility=False)
    assert res_infeasible.is_feasible is False


def test_automated_triage_pruning(wind: WindField) -> None:
    """Verify triage logic prunes low-priority targets to yield a feasible sortie."""
    gen = SyntheticScenarioGenerator()
    scenario = gen.generate(num_survivors=8, wind_speed_mps=wind.speed_mps, wind_direction_deg=wind.direction_deg, seed=42)
    waypoints = scenario.waypoints

    # Drone with tight battery that can only complete a partial tour
    restricted_drone = DronePhysicsModel(battery_capacity_wh=50.0, reserve_margin_ratio=0.15)
    calc = CostMatrixCalculator(restricted_drone, wind)
    cost_res = calc.compute(waypoints)

    dp_solver = BitmaskDPSolver(restricted_drone)
    triage_res = dp_solver.solve(cost_res, waypoints, enforce_feasibility=True)

    # Triage should be applied
    assert triage_res.triage_applied is True
    assert triage_res.is_feasible is True
    assert triage_res.total_energy_wh <= restricted_drone.usable_energy_wh


# ==============================================================================
# PHASE 4 TESTS: MAVLink Telemetry Export
# ==============================================================================

def test_mavlink_waypoint_file_generation() -> None:
    """Verify QGroundControl WPL 110 formatted waypoint generation."""
    gen = SyntheticScenarioGenerator()
    scenario = gen.generate(num_survivors=4, seed=42)
    waypoints = scenario.waypoints

    telemetry_gen = MAVLinkMissionGenerator(
        survey_alt_agl_m=50.0,
        loiter_time_s=5.0,
        acceptance_radius_m=3.0,
    )

    path = [0, 2, 1, 3, 4, 0]
    wp_text = telemetry_gen.generate_waypoints_text(path, waypoints, include_rtl=True)

    # Must start with standard header
    assert wp_text.startswith("QGC WPL 110\n")

    # Validate syntax using built-in validator
    assert MAVLinkMissionGenerator.validate_waypoints_text(wp_text) is True

    lines = [line for line in wp_text.strip().split("\n") if line]
    # Header (1) + Home (1) + Takeoff (1) + 5 waypoints + RTL (1) = 9 lines
    assert len(lines) == 9

    # Last line should be RTL (command code 20)
    rtl_parts = lines[-1].split("\t")
    assert rtl_parts[3] == "20"
