"""
telemetry_exporter.py - MAVLink / QGroundControl Mission Exporter.

Converts 2D Cartesian local trajectory coordinates into WGS84 Geodetic coordinates
and generates standard QGroundControl WPL 110 mission waypoint files (.waypoints).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Union


# WGS84 Constants
WGS84_A: float = 6378137.0  # Equatorial radius in meters
WGS84_F: float = 1.0 / 298.257223563  # Flattening
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # Polar semi-minor axis

# Standard MAVLink Command Codes (MAV_CMD)
MAV_CMD_NAV_WAYPOINT: int = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH: int = 20
MAV_CMD_NAV_LAND: int = 21
MAV_CMD_NAV_TAKEOFF: int = 22

# Coordinate Reference Frames (MAV_FRAME)
MAV_FRAME_GLOBAL: int = 0
MAV_FRAME_GLOBAL_RELATIVE_ALT: int = 3


class MAVLinkMissionExporter:
    """Exports autonomous UAV flight routes to standard QGroundControl WPL 110 format."""

    def __init__(
        self,
        home_coords: Tuple[float, float, float],
        survey_alt_agl_m: float = 30.0,
        loiter_time_s: float = 5.0,
        acceptance_radius_m: float = 3.0,
    ) -> None:
        """Initialize exporter with base launch datum and flight parameters.

        Args:
            home_coords: Tuple of (latitude, longitude, altitude_msl).
            survey_alt_agl_m: Target survey altitude Above Ground Level (meters).
            loiter_time_s: Camera survey / loiter duration at survivor target (seconds).
            acceptance_radius_m: Navigation waypoint arrival threshold (meters).
        """
        self.home_lat, self.home_lon, self.home_alt = home_coords
        self.survey_alt_agl_m = survey_alt_agl_m
        self.loiter_time_s = loiter_time_s
        self.acceptance_radius_m = acceptance_radius_m

    def cartesian_to_wgs84(self, dx: float, dy: float) -> Tuple[float, float]:
        """Convert local Cartesian East-North offsets (meters) to WGS84 GPS (lat, lon).

        Uses local tangent plane flat-earth projection accurate within tactical SAR bounds (< 50 km).

        Args:
            dx: East displacement in meters relative to home datum.
            dy: North displacement in meters relative to home datum.

        Returns:
            Tuple of (latitude, longitude) in decimal degrees.
        """
        # Latitude displacement
        delta_lat = (dy / WGS84_A) * (180.0 / math.pi)

        # Longitude displacement scaled by cosine of latitude
        lat_rad = math.radians(self.home_lat)
        cos_lat = max(math.cos(lat_rad), 1e-6)
        delta_lon = (dx / (WGS84_A * cos_lat)) * (180.0 / math.pi)

        target_lat = self.home_lat + delta_lat
        target_lon = self.home_lon + delta_lon
        return target_lat, target_lon

    def generate_wpl110_text(
        self,
        waypoints_sequence: List[Tuple[float, float]],
        include_rtl: bool = True,
    ) -> str:
        """Construct official QGroundControl WPL 110 mission file text.

        Layout protocol per line:
            <INDEX> <CURRENT> <FRAME> <COMMAND> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <LAT> <LON> <ALT> <AUTOCONTINUE>

        Sequence:
            - Sequence 0: Home location (MAV_CMD_NAV_WAYPOINT)
            - Sequence 1: Takeoff to survey altitude 30 m (MAV_CMD_NAV_TAKEOFF)
            - Sequence 2 to n+1: Survey transit to each survivor waypoint (MAV_CMD_NAV_WAYPOINT)
            - Sequence n+2: Return to Launch (MAV_CMD_NAV_RETURN_TO_LAUNCH)
        """
        lines: List[str] = ["QGC WPL 110"]
        seq = 0

        # Sequence 0: Home Location Reference
        lines.append(
            f"{seq}\t1\t{MAV_FRAME_GLOBAL}\t{MAV_CMD_NAV_WAYPOINT}\t"
            f"0.000000\t0.000000\t0.000000\t0.000000\t"
            f"{self.home_lat:.7f}\t{self.home_lon:.7f}\t{self.home_alt:.1f}\t1"
        )
        seq += 1

        # Sequence 1: Autonomous Takeoff Command to survey altitude (30m)
        lines.append(
            f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_TAKEOFF}\t"
            f"0.000000\t0.000000\t0.000000\t0.000000\t"
            f"{self.home_lat:.7f}\t{self.home_lon:.7f}\t{self.survey_alt_agl_m:.1f}\t1"
        )
        seq += 1

        # Sequence 2 to n+1: Transit to each survivor target
        # Filter out redundant origin (0, 0) if at start of sequence
        tour = waypoints_sequence[1:] if waypoints_sequence and waypoints_sequence[0] == (0.0, 0.0) else waypoints_sequence

        for wp in tour:
            dx, dy = wp
            lat, lon = self.cartesian_to_wgs84(dx, dy)
            is_base = math.isclose(dx, 0.0, abs_tol=1e-3) and math.isclose(dy, 0.0, abs_tol=1e-3)
            loiter = 0.0 if is_base else self.loiter_time_s

            lines.append(
                f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
                f"{loiter:.1f}\t{self.acceptance_radius_m:.2f}\t0.000000\t0.000000\t"
                f"{lat:.7f}\t{lon:.7f}\t{self.survey_alt_agl_m:.1f}\t1"
            )
            seq += 1

        # Sequence n+2: Return to Launch (RTL)
        if include_rtl:
            lines.append(
                f"{seq}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_RETURN_TO_LAUNCH}\t"
                f"0.000000\t0.000000\t0.000000\t0.000000\t"
                f"0.000000\t0.000000\t0.000000\t1"
            )

        return "\n".join(lines) + "\n"

    def export_mission(
        self,
        waypoints_sequence: List[Tuple[float, float]],
        filename: Union[str, Path] = "mission.waypoints",
        include_rtl: bool = True,
    ) -> str:
        """Export the mission text to a .waypoints file on disk.

        Args:
            waypoints_sequence: Ordered list of (x, y) coordinates.
            filename: Destination filepath for the QGC WPL 110 mission.
            include_rtl: Whether to append RTL command at the end.

        Returns:
            Resolved absolute path to the generated mission file.
        """
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate_wpl110_text(waypoints_sequence, include_rtl=include_rtl)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(out_path.resolve())
