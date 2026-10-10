"""
core/telemetry.py - MAVLink & QGroundControl Mission Exporter.

Converts optimized cyber-physical UAV trajectories into industry-standard
QGroundControl WPL 110 mission waypoint files (.waypoints) compatible with
PX4 Autopilot, ArduPilot, and Mission Planner.
"""

from __future__ import annotations

import os
from typing import List, Optional
from pathlib import Path

from .physics import Waypoint


# Standard MAVLink Command Codes (MAV_CMD)
MAV_CMD_NAV_WAYPOINT: int = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH: int = 20
MAV_CMD_NAV_LAND: int = 21
MAV_CMD_NAV_TAKEOFF: int = 22

# Standard Coordinate Frames (MAV_FRAME)
MAV_FRAME_GLOBAL: int = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT: int = 3


class MAVLinkMissionGenerator:
    """Exports autonomous search-and-rescue routes to QGroundControl WPL 110 format."""

    def __init__(
        self,
        survey_alt_agl_m: float = 50.0,
        loiter_time_s: float = 5.0,
        acceptance_radius_m: float = 3.0,
    ) -> None:
        self.survey_alt_agl_m = survey_alt_agl_m
        self.loiter_time_s = loiter_time_s
        self.acceptance_radius_m = acceptance_radius_m

    def generate_waypoints_text(
        self,
        optimal_path: List[int],
        waypoints: List[Waypoint],
        include_rtl: bool = True,
    ) -> str:
        """Construct the QGroundControl WPL 110 formatted string.

        Protocol layout:
            INDEX <tab> CURRENT_WP <tab> COORD_FRAME <tab> COMMAND <tab>
            PARAM1 <tab> PARAM2 <tab> PARAM3 <tab> PARAM4 <tab>
            LATITUDE <tab> LONGITUDE <tab> ALTITUDE <tab> AUTOCONTINUE

        Sequence:
            0: Home / Launch reference coordinate (MAV_CMD_NAV_WAYPOINT)
            1: Takeoff climb to survey altitude (MAV_CMD_NAV_TAKEOFF)
            2..K: Sequential survivor waypoints (MAV_CMD_NAV_WAYPOINT)
            K+1: Return to Launch (MAV_CMD_NAV_RETURN_TO_LAUNCH) or Land
        """
        if not optimal_path or len(waypoints) == 0:
            raise ValueError("Cannot generate mission from empty route or waypoints.")

        lines: List[str] = ["QGC WPL 110"]
        idx = 0

        # Base depot (always index 0 of waypoints)
        depot = waypoints[0]

        # Item 0: Home Location Reference
        lines.append(
            f"{idx}\t1\t{MAV_FRAME_GLOBAL}\t{MAV_CMD_NAV_WAYPOINT}\t"
            f"0.000000\t0.000000\t0.000000\t0.000000\t"
            f"{depot.lat:.7f}\t{depot.lon:.7f}\t{depot.alt_agl_m:.1f}\t1"
        )
        idx += 1

        # Item 1: Autonomous Takeoff Command
        lines.append(
            f"{idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_TAKEOFF}\t"
            f"0.000000\t0.000000\t0.000000\t0.000000\t"
            f"{depot.lat:.7f}\t{depot.lon:.7f}\t{self.survey_alt_agl_m:.1f}\t1"
        )
        idx += 1

        # Items 2..N: Sequential Waypoints in the tour (skipping the initial depot 0)
        # We iterate over path[1:]
        tour_targets = optimal_path[1:]
        for node_idx in tour_targets:
            wp = waypoints[node_idx]

            if node_idx == 0:
                # Returning to Base Depot
                lines.append(
                    f"{idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
                    f"0.000000\t{self.acceptance_radius_m:.2f}\t0.000000\t0.000000\t"
                    f"{wp.lat:.7f}\t{wp.lon:.7f}\t{self.survey_alt_agl_m:.1f}\t1"
                )
            else:
                # Survivor triage beacon survey point
                lines.append(
                    f"{idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
                    f"{wp.loiter_time_s:.1f}\t{self.acceptance_radius_m:.2f}\t0.000000\t0.000000\t"
                    f"{wp.lat:.7f}\t{wp.lon:.7f}\t{wp.alt_agl_m:.1f}\t1"
                )
            idx += 1

        # Final command: Return To Launch (RTL)
        if include_rtl:
            lines.append(
                f"{idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_RETURN_TO_LAUNCH}\t"
                f"0.000000\t0.000000\t0.000000\t0.000000\t"
                f"0.000000\t0.000000\t0.000000\t1"
            )

        return "\n".join(lines) + "\n"

    def export_to_file(
        self,
        optimal_path: List[int],
        waypoints: List[Waypoint],
        output_filepath: str | Path,
        include_rtl: bool = True,
    ) -> str:
        """Write QGroundControl waypoints to disk and return the path."""
        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate_waypoints_text(
            optimal_path=optimal_path,
            waypoints=waypoints,
            include_rtl=include_rtl,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(out_path.resolve())

    @staticmethod
    def validate_waypoints_text(content: str) -> bool:
        """Verify structural validity and syntax of a QGC WPL 110 mission file."""
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        if not lines or lines[0] != "QGC WPL 110":
            return False

        for i, line in enumerate(lines[1:]):
            parts = line.split("\t")
            if len(parts) != 12:
                return False
            # Verify index sequencing
            if int(parts[0]) != i:
                return False
            # Verify coordinates are valid floats
            lat = float(parts[8])
            lon = float(parts[9])
            # Valid coordinate ranges or 0.0 for RTL
            cmd = int(parts[3])
            if cmd != MAV_CMD_NAV_RETURN_TO_LAUNCH:
                if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                    return False

        return True
