"""
tests/test_bridge.py - Unit & Integration Tests for Evaluation, Telemetry & Simulation Bridge.

Tests:
1. MAVLinkMissionExporter: Cartesian-to-WGS84 projection and QGC WPL 110 mission structure.
2. AeroSARBenchmarkRunner: 5 km x 5 km disaster zone generator and academic table output.
3. Trajectory visualizer: Side-by-side plot generation with clean white background.
"""

import math
from pathlib import Path
import pytest

from telemetry_exporter import MAVLinkMissionExporter
from benchmark_runner import AeroSARBenchmarkRunner
from trajectory_visualizer import plot_trajectory_comparison


def test_mavlink_exporter_cartesian_to_wgs84() -> None:
    """Verify flat-earth to geodetic WGS84 GPS coordinate conversion."""
    home = (46.558000, 8.560000, 1450.0)
    exporter = MAVLinkMissionExporter(home_coords=home, survey_alt_agl_m=30.0)

    # 1000m North (dy=1000, dx=0)
    lat_n, lon_n = exporter.cartesian_to_wgs84(0.0, 1000.0)
    assert lat_n > home[0]
    assert math.isclose(lon_n, home[1], abs_tol=1e-5)

    # 1000m East (dy=0, dx=1000)
    lat_e, lon_e = exporter.cartesian_to_wgs84(1000.0, 0.0)
    assert math.isclose(lat_e, home[0], abs_tol=1e-5)
    assert lon_e > home[1]


def test_mavlink_exporter_wpl110_format(tmp_path: Path) -> None:
    """Verify QGroundControl WPL 110 file structure with takeoff, waypoints, and RTL."""
    home = (46.558000, 8.560000, 1450.0)
    exporter = MAVLinkMissionExporter(home_coords=home, survey_alt_agl_m=30.0, loiter_time_s=5.0)

    coords = [(0.0, 0.0), (500.0, 500.0), (-300.0, 800.0), (0.0, 0.0)]
    wpl_text = exporter.generate_wpl110_text(coords, include_rtl=True)

    lines = [ln.strip() for ln in wpl_text.strip().split("\n") if ln.strip()]
    assert lines[0] == "QGC WPL 110"

    # Line 1: Sequence 0 Home (Command 16)
    seq0 = lines[1].split("\t")
    assert seq0[0] == "0"
    assert seq0[3] == "16"
    assert math.isclose(float(seq0[8]), home[0], abs_tol=1e-5)

    # Line 2: Sequence 1 Takeoff to 30m (Command 22)
    seq1 = lines[2].split("\t")
    assert seq1[0] == "1"
    assert seq1[3] == "22"
    assert float(seq1[10]) == 30.0

    # Line 3 & 4: Sequence 2 & 3 Waypoints (Command 16)
    seq2 = lines[3].split("\t")
    assert seq2[0] == "2"
    assert seq2[3] == "16"
    assert float(seq2[4]) == 5.0  # Loiter time 5s
    assert float(seq2[10]) == 30.0  # Survey alt 30m

    # Last line: Sequence 5 RTL (Command 20)
    last_line = lines[-1].split("\t")
    assert last_line[3] == "20"

    # Export to disk test
    out_file = tmp_path / "test_mission.waypoints"
    exported_path = exporter.export_mission(coords, filename=out_file)
    assert Path(exported_path).exists()


def test_benchmark_runner_disaster_zone_bounds() -> None:
    """Verify generated survivor positions stay within 5 km x 5 km bounding area."""
    runner = AeroSARBenchmarkRunner(bounding_area_km=5.0, seed=42)
    wps = runner.generate_disaster_zone(10)

    assert len(wps) == 11
    assert wps[0].x == 0.0 and wps[0].y == 0.0  # Base depot at origin

    for wp in wps[1:]:
        assert -2500.0 <= wp.x <= 2500.0
        assert -2500.0 <= wp.y <= 2500.0


def test_benchmark_runner_dp_brute_force_match() -> None:
    """Verify DP matches Brute Force on small problem instances."""
    runner = AeroSARBenchmarkRunner(bounding_area_km=5.0, seed=123)
    records, table = runner.run_benchmark(target_sizes=[4, 6])

    assert len(records) == 2
    for r in records:
        assert r.dp_matches_brute_force is True
        assert r.dp_energy_wh > 0.0
        assert r.is_feasible is True
    assert "| **4** |" in table
    assert "| **6** |" in table


def test_trajectory_visualizer_rendering(tmp_path: Path) -> None:
    """Verify trajectory visualizer produces publication image cleanly."""
    out_img = tmp_path / "test_analysis.png"
    coords = [(0.0, 0.0), (1000.0, 500.0), (-500.0, 1200.0), (800.0, -900.0)]
    labels = ["BASE", "S1", "S2", "S3"]
    dp_path = [0, 1, 2, 3, 0]
    greedy_path = [0, 3, 1, 2, 0]

    fig = plot_trajectory_comparison(
        waypoints=coords,
        waypoint_labels=labels,
        dp_path=dp_path,
        greedy_path=greedy_path,
        wind_vector=(4.24, 4.24),
        dp_energy_wh=65.0,
        greedy_energy_wh=75.0,
        dp_flight_time_s=600.0,
        greedy_flight_time_s=680.0,
        speedup_factor=50.0,
        is_feasible=True,
        save_path=out_img,
    )

    assert out_img.exists()
    assert out_img.stat().st_size > 10000  # Valid non-empty PNG image
