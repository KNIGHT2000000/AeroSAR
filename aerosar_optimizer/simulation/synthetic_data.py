"""
simulation/synthetic_data.py - Synthetic Survivor Beacon & Wind Field Generator.

Generates realistic search-and-rescue (SAR) operational scenarios featuring:
1. Clustered and dispersed survivor beacon locations with triage urgency ratings.
2. 2D vector wind fields with directional shear and gust profiles.
3. Accurate Geodetic WGS84 and Cartesian Euclidean coordinate mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.physics import Waypoint, WindField, DronePhysicsModel


@dataclass(slots=True)
class SARScenario:
    """Contains complete environmental and target parameters for a SAR sortie.

    Attributes:
        name: Tactical scenario codename.
        datum_lat: Base depot reference latitude (degrees).
        datum_lon: Base depot reference longitude (degrees).
        datum_alt_m: Base depot launch altitude MSL (meters).
        waypoints: List of Waypoints (index 0 is depot, indices 1..N-1 are survivors).
        wind_field: Atmospheric 2D wind field vector.
        operational_radius_m: Outer bounding radius of search zone (meters).
    """
    name: str
    datum_lat: float
    datum_lon: float
    datum_alt_m: float
    waypoints: List[Waypoint]
    wind_field: WindField
    operational_radius_m: float

    @property
    def num_survivors(self) -> int:
        """Total number of survivor beacon targets."""
        return max(0, len(self.waypoints) - 1)

    @property
    def survivor_waypoints(self) -> List[Waypoint]:
        """List of survivor beacons excluding base depot."""
        return self.waypoints[1:]


class SyntheticScenarioGenerator:
    """Factory generating clustered, realistic SAR target distributions and wind environments."""

    def __init__(
        self,
        datum_lat: float = 46.558000,
        datum_lon: float = 8.560000,
        datum_alt_m: float = 1450.0,
        search_radius_m: float = 2500.0,
    ) -> None:
        self.datum_lat = datum_lat
        self.datum_lon = datum_lon
        self.datum_alt_m = datum_alt_m
        self.search_radius_m = search_radius_m

    def generate(
        self,
        num_survivors: int = 10,
        wind_speed_mps: float = 6.8,
        wind_direction_deg: float = 55.0,
        scenario_preset: str = "ALPINE_AVALANCHE",
        seed: Optional[int] = 42,
    ) -> SARScenario:
        """Create a synthetic SAR scenario with designated survivor clusters and wind.

        Args:
            num_survivors: Number of survivor beacon targets (n).
            wind_speed_mps: Ambient wind speed magnitude in m/s.
            wind_direction_deg: Wind bearing angle (degrees from East).
            scenario_preset: Distribution archetype:
                - 'ALPINE_AVALANCHE': Clustered debris zones along mountain gully.
                - 'WILDERNESS_DISPERSED': Broadly scattered wilderness groups.
                - 'COASTAL_MARITIME': Elongated coastal search drift patterns.
            seed: RNG seed for reproducible experiments.

        Returns:
            SARScenario containing the Base Depot and survivor waypoints.
        """
        rng = np.random.default_rng(seed)

        # Base Depot at Cartesian origin (0, 0)
        depot = Waypoint.from_cartesian(
            id_=0,
            label="BASE_DEPOT",
            x=0.0,
            y=0.0,
            datum_lat=self.datum_lat,
            datum_lon=self.datum_lon,
            priority=0.0,
            loiter_time_s=0.0,
            alt_agl_m=0.0,
        )

        waypoints: List[Waypoint] = [depot]

        if scenario_preset == "ALPINE_AVALANCHE":
            # 2 to 3 avalanche gully clusters + isolated scouts
            num_clusters = min(3, max(1, num_survivors // 3))
            cluster_centers = []
            for _ in range(num_clusters):
                # Centers distributed 600m to 1800m from base
                r = rng.uniform(600.0, self.search_radius_m * 0.75)
                theta = rng.uniform(0.0, 2.0 * math.pi)
                cluster_centers.append((r * math.cos(theta), r * math.sin(theta)))

            for i in range(1, num_survivors + 1):
                c_idx = (i - 1) % len(cluster_centers)
                cx, cy = cluster_centers[c_idx]

                # Gaussian scatter around cluster center (std = 180m)
                offset_x = rng.normal(0.0, 180.0)
                offset_y = rng.normal(0.0, 180.0)
                px = np.clip(cx + offset_x, -self.search_radius_m, self.search_radius_m)
                py = np.clip(cy + offset_y, -self.search_radius_m, self.search_radius_m)

                # Priority: 1.0 (mild) to 10.0 (critical hypothermia)
                priority = float(np.round(rng.uniform(2.0, 10.0), 1))
                loiter_s = float(rng.choice([4.0, 5.0, 6.0, 8.0]))

                wp = Waypoint.from_cartesian(
                    id_=i,
                    label=f"SURVIVOR_{i:02d}",
                    x=px,
                    y=py,
                    datum_lat=self.datum_lat,
                    datum_lon=self.datum_lon,
                    priority=priority,
                    loiter_time_s=loiter_s,
                    alt_agl_m=50.0,
                )
                waypoints.append(wp)

        elif scenario_preset == "COASTAL_MARITIME":
            # Linear drift line along ocean current / coastline
            shore_angle = math.radians(35.0)
            for i in range(1, num_survivors + 1):
                drift_dist = rng.uniform(-self.search_radius_m * 0.8, self.search_radius_m * 0.8)
                cross_dist = rng.normal(0.0, 300.0)

                px = drift_dist * math.cos(shore_angle) - cross_dist * math.sin(shore_angle)
                py = drift_dist * math.sin(shore_angle) + cross_dist * math.cos(shore_angle)

                priority = float(np.round(rng.uniform(3.0, 9.5), 1))
                wp = Waypoint.from_cartesian(
                    id_=i,
                    label=f"MARITIME_BEACON_{i:02d}",
                    x=px,
                    y=py,
                    datum_lat=self.datum_lat,
                    datum_lon=self.datum_lon,
                    priority=priority,
                    loiter_time_s=5.0,
                    alt_agl_m=40.0,
                )
                waypoints.append(wp)

        else:  # WILDERNESS_DISPERSED
            # Uniform disk distribution
            for i in range(1, num_survivors + 1):
                r = math.sqrt(rng.uniform(0.1, 1.0)) * (self.search_radius_m * 0.9)
                theta = rng.uniform(0.0, 2.0 * math.pi)
                px = r * math.cos(theta)
                py = r * math.sin(theta)

                priority = float(np.round(rng.uniform(1.0, 10.0), 1))
                wp = Waypoint.from_cartesian(
                    id_=i,
                    label=f"LOST_HIKER_{i:02d}",
                    x=px,
                    y=py,
                    datum_lat=self.datum_lat,
                    datum_lon=self.datum_lon,
                    priority=priority,
                    loiter_time_s=5.0,
                    alt_agl_m=50.0,
                )
                waypoints.append(wp)

        wind = WindField(speed_mps=wind_speed_mps, direction_deg=wind_direction_deg)

        return SARScenario(
            name=f"SAR-{scenario_preset}-N{num_survivors}",
            datum_lat=self.datum_lat,
            datum_lon=self.datum_lon,
            datum_alt_m=self.datum_alt_m,
            waypoints=waypoints,
            wind_field=wind,
            operational_radius_m=self.search_radius_m,
        )
