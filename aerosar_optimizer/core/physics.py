"""
core/physics.py - Kinematic Drone Flight Physics, Wind Drift & Battery Model.

Models cyber-physical UAV aerodynamics under steady/turbulent wind fields:
1. Waypoint entity in local Cartesian (x, y) & WGS84 (lat, lon) coordinate frames.
2. 2D wind field vector representation V_wind = (V_wx, V_wy).
3. Wind-triangle kinematic drift resolution solving for effective ground speed V_ground.
4. Non-linear quadcopter electrical power consumption:
     P_flight = P_hover + c_drag * ||V_air||^3
5. Directed energy computation from node i to node j:
     C[i][j] = P_flight * (||X_j - X_i|| / ||V_ground||) + P_hover * t_loiter
6. Battery pack capacity, reserve margin, and State-of-Charge (SoC) tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


# WGS84 Earth radius constant (mean radius in meters)
R_EARTH_METERS: float = 6371000.0


@dataclass(slots=True)
class Waypoint:
    """Represents a 2D Euclidean / Geodetic navigation waypoint.

    Attributes:
        id: Unique integer index (0 is typically reserved for base depot).
        label: Human-readable identifier (e.g., 'DEPOT', 'SURVIVOR_01').
        x: Local Cartesian East displacement from datum in meters.
        y: Local Cartesian North displacement from datum in meters.
        lat: WGS84 Latitude in decimal degrees.
        lon: WGS84 Longitude in decimal degrees.
        priority: Triage priority score (e.g., 1.0 to 10.0, higher is more urgent).
        loiter_time_s: Required survey/sensor loiter time at waypoint (seconds).
        alt_agl_m: Target survey altitude above ground level in meters.
    """
    id: int
    label: str
    x: float
    y: float
    lat: float = 0.0
    lon: float = 0.0
    priority: float = 1.0
    loiter_time_s: float = 5.0
    alt_agl_m: float = 50.0

    @property
    def position_vec(self) -> np.ndarray:
        """2D position vector [x, y] in meters."""
        return np.array([self.x, self.y], dtype=np.float64)

    def distance_to(self, other: Waypoint) -> float:
        """Compute Euclidean distance in meters to another waypoint."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.hypot(dx, dy)

    @classmethod
    def from_cartesian(
        cls,
        id_: int,
        label: str,
        x: float,
        y: float,
        datum_lat: float,
        datum_lon: float,
        priority: float = 1.0,
        loiter_time_s: float = 5.0,
        alt_agl_m: float = 50.0,
    ) -> Waypoint:
        """Instantiate Waypoint from local Cartesian offsets relative to a base datum.

        Uses flat-earth projection suitable for tactical UAV SAR zones (< 50 km).
        """
        # Convert Cartesian delta (meters) to angular displacement (degrees)
        d_lat_deg = (y / R_EARTH_METERS) * (180.0 / math.pi)
        lat_rad = math.radians(datum_lat)
        cos_lat = math.cos(lat_rad)
        # Avoid division by zero at poles
        cos_lat = max(cos_lat, 1e-6)
        d_lon_deg = (x / (R_EARTH_METERS * cos_lat)) * (180.0 / math.pi)

        return cls(
            id=id_,
            label=label,
            x=float(x),
            y=float(y),
            lat=datum_lat + d_lat_deg,
            lon=datum_lon + d_lon_deg,
            priority=priority,
            loiter_time_s=loiter_time_s,
            alt_agl_m=alt_agl_m,
        )


@dataclass(slots=True)
class WindField:
    """Represents a 2D horizontal atmospheric wind field vector.

    Attributes:
        speed_mps: Wind velocity magnitude in m/s.
        direction_deg: Meteorological direction in degrees (0° = East, 90° = North).
                       (Angle vector points TOWARDS).
        vx: Wind vector X-component (Eastwards) in m/s.
        vy: Wind vector Y-component (Northwards) in m/s.
    """
    speed_mps: float
    direction_deg: float
    vx: float = field(init=False)
    vy: float = field(init=False)

    def __post_init__(self) -> None:
        rad = math.radians(self.direction_deg)
        object.__setattr__(self, "vx", self.speed_mps * math.cos(rad))
        object.__setattr__(self, "vy", self.speed_mps * math.sin(rad))

    @property
    def vector(self) -> np.ndarray:
        """2D wind vector [vx, vy] in m/s."""
        return np.array([self.vx, self.vy], dtype=np.float64)

    @classmethod
    def from_components(cls, vx: float, vy: float) -> WindField:
        """Create WindField from Cartesian velocity components."""
        speed = math.hypot(vx, vy)
        deg = math.degrees(math.atan2(vy, vx)) % 360.0
        return cls(speed_mps=speed, direction_deg=deg)


@dataclass(slots=True)
class DronePhysicsModel:
    """Aerodynamic, kinematic, and electrical power consumption model for a multirotor UAV.

    Attributes:
        nominal_airspeed_mps: Commanded true airspeed magnitude V_air in m/s.
        max_airspeed_mps: Upper aerodynamic airspeed ceiling in m/s.
        hover_power_watts: Electrical power consumed during steady hover (Watts).
        drag_coefficient: Lumped parasitic aerodynamic drag coefficient c_drag (W / (m/s)^3).
        battery_capacity_wh: Nominal battery pack capacity in Watt-hours.
        reserve_margin_ratio: Minimum battery buffer ratio reserved for emergency RTL (e.g. 0.20 for 20%).
        nominal_voltage_v: Nominal battery pack voltage (Volts).
    """
    nominal_airspeed_mps: float = 15.0
    max_airspeed_mps: float = 24.0
    hover_power_watts: float = 310.0
    drag_coefficient: float = 0.048
    battery_capacity_wh: float = 350.0
    reserve_margin_ratio: float = 0.20
    nominal_voltage_v: float = 22.2

    @property
    def total_energy_joules(self) -> float:
        """Total battery pack energy capacity in Joules (1 Wh = 3600 J)."""
        return self.battery_capacity_wh * 3600.0

    @property
    def usable_energy_joules(self) -> float:
        """Usable mission energy envelope after subtracting reserve margin (Joules)."""
        return self.total_energy_joules * (1.0 - self.reserve_margin_ratio)

    @property
    def usable_energy_wh(self) -> float:
        """Usable mission energy envelope in Watt-hours."""
        return self.battery_capacity_wh * (1.0 - self.reserve_margin_ratio)

    def power_consumption(self, airspeed_mps: Optional[float] = None) -> float:
        """Calculate electrical power consumption in Watts.

        Non-linear aerodynamic drag model:
            P_flight = P_hover + c_drag * ||V_air||^3
        """
        v_air = self.nominal_airspeed_mps if airspeed_mps is None else airspeed_mps
        return self.hover_power_watts + self.drag_coefficient * (v_air ** 3)

    def solve_wind_triangle(
        self,
        displacement_vec: np.ndarray,
        wind_field: WindField,
    ) -> Tuple[float, np.ndarray, float]:
        """Solve the kinematic wind triangle to determine effective ground speed.

        For a flight leg along ground displacement D = X_j - X_i:
            Unit ground direction: u_hat = D / ||D||
            Kinematic vector constraint: V_ground = V_air_vec + V_wind
            Along course: V_ground = v_g * u_hat (with v_g > 0)
            Thus: V_air_vec = v_g * u_hat - V_wind

        Given commanded airspeed magnitude ||V_air_vec|| = V_air:
            ||v_g * u_hat - V_wind||^2 = V_air^2
            v_g^2 - 2 * (u_hat . V_wind) * v_g + (||V_wind||^2 - V_air^2) = 0

        Let:
            w_par = u_hat . V_wind (parallel wind component)
            w_perp_sq = ||V_wind||^2 - w_par^2 (squared crosswind component)
            Discriminant = V_air^2 - w_perp_sq

        Returns:
            Tuple of:
                ground_speed (float): Magnitude of ground velocity v_g in m/s.
                ground_velocity_vec (np.ndarray): 2D ground velocity vector [vg_x, vg_y].
                crab_angle_rad (float): Heading correction crab angle in radians.

        Raises:
            ValueError: If crosswind exceeds maximum airspeed (flight impossible).
            ValueError: If headwind prevents positive forward ground speed.
        """
        dist = float(np.linalg.norm(displacement_vec))
        if dist < 1e-6:
            # Zero displacement (stay at same node)
            return self.nominal_airspeed_mps, np.zeros(2, dtype=np.float64), 0.0

        u_hat = displacement_vec / dist
        v_w = wind_field.vector
        v_air = self.nominal_airspeed_mps

        # Components of wind relative to desired ground track
        w_parallel = float(np.dot(u_hat, v_w))
        v_w_mag_sq = float(np.dot(v_w, v_w))
        w_perp_sq = max(0.0, v_w_mag_sq - (w_parallel ** 2))

        discriminant = (v_air ** 2) - w_perp_sq
        if discriminant < 0.0:
            raise ValueError(
                f"Aerodynamic stall/infeasibility: Crosswind component "
                f"({math.sqrt(w_perp_sq):.2f} m/s) exceeds drone airspeed ({v_air:.2f} m/s)."
            )

        # Ground speed magnitude solving quadratic equation
        v_ground = w_parallel + math.sqrt(discriminant)

        if v_ground <= 0.1:
            raise ValueError(
                f"Headwind saturation: Ground speed ({v_ground:.2f} m/s) is non-positive. "
                f"Drone cannot advance along bearing."
            )

        ground_vel_vec = v_ground * u_hat
        w_perp = math.sqrt(w_perp_sq)
        crab_angle = math.asin(min(1.0, max(-1.0, w_perp / v_air)))

        return v_ground, ground_vel_vec, crab_angle

    def calculate_leg_cost(
        self,
        origin: Waypoint,
        target: Waypoint,
        wind_field: WindField,
    ) -> Tuple[float, float, float]:
        """Compute directed energy cost, flight duration, and ground speed from origin to target.

        Energy cost formulation:
            t_flight = distance / v_ground
            E_flight = P_flight * t_flight
            E_loiter = P_hover * target.loiter_time_s
            C[i][j] = E_flight + E_loiter (in Joules)

        Returns:
            Tuple of:
                energy_joules (float): Total energy consumed on this leg.
                flight_time_s (float): Travel duration plus loiter time in seconds.
                ground_speed_mps (float): Effective ground speed in m/s.
        """
        if origin.id == target.id:
            return 0.0, 0.0, self.nominal_airspeed_mps

        disp = target.position_vec - origin.position_vec
        dist = float(np.linalg.norm(disp))

        try:
            v_ground, _, _ = self.solve_wind_triangle(disp, wind_field)
        except ValueError:
            # Unreachable due to wind barrier
            return float("inf"), float("inf"), 0.0

        p_flight = self.power_consumption(self.nominal_airspeed_mps)
        t_transit = dist / v_ground
        e_transit = p_flight * t_transit

        # Loiter at target waypoint for sensor thermal imaging / survivor assessment
        t_loiter = target.loiter_time_s
        e_loiter = self.hover_power_watts * t_loiter

        total_energy = e_transit + e_loiter
        total_time = t_transit + t_loiter

        return total_energy, total_time, v_ground

    def state_of_charge_percent(self, energy_consumed_joules: float) -> float:
        """Compute remaining battery State-of-Charge (SoC) as a percentage [0..100]."""
        rem = max(0.0, self.total_energy_joules - energy_consumed_joules)
        return (rem / self.total_energy_joules) * 100.0

    def is_mission_feasible(self, total_energy_joules: float) -> bool:
        """Validate if the energy consumed respects the usable battery limit."""
        return bool(total_energy_joules <= self.usable_energy_joules)
