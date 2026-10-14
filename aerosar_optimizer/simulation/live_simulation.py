"""
simulation/live_simulation.py - Cyber-Physical Live UAV Flight Kinematics & Dual-Drone Race Engine.

Simulates continuous real-time multirotor UAV trajectory dynamics under steady atmospheric wind drift:
1. Continuous position interpolation X(t) = [x(t), y(t)] with wind-triangle heading resolution.
2. Waypoint loitering/surveying simulation at each survivor beacon.
3. Electrical power integration P(t) tracking dynamic Battery State of Charge (SoC %).
4. Synchronized dual-drone flight race engine comparing Proposed Bitmask DP vs. Traditional Baseline.
5. Exportable telemetry frames for 60 FPS interactive HTML5 Radar HUD and Plotly graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from core.physics import DronePhysicsModel, WindField, Waypoint
from core.bitmask_dp import OptimizationResult


@dataclass(slots=True)
class DroneFlightSnapshot:
    """Instantaneous aerodynamic and kinematic telemetry state of a UAV at time t."""
    time_s: float
    x: float
    y: float
    lat: float
    lon: float
    heading_deg: float
    ground_speed_mps: float
    airspeed_mps: float
    current_leg_idx: int
    from_node_id: int
    to_node_id: int
    target_label: str
    status: str
    power_watts: float
    energy_consumed_wh: float
    energy_consumed_joules: float
    soc_percent: float
    distance_flown_m: float
    visited_nodes: List[int]
    is_loitering: bool
    is_landed: bool
    is_feasible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": round(self.time_s, 2),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "heading_deg": round(self.heading_deg, 1),
            "ground_speed_mps": round(self.ground_speed_mps, 2),
            "airspeed_mps": round(self.airspeed_mps, 2),
            "current_leg_idx": self.current_leg_idx,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "target_label": self.target_label,
            "status": self.status,
            "power_watts": round(self.power_watts, 1),
            "energy_consumed_wh": round(self.energy_consumed_wh, 2),
            "soc_percent": round(self.soc_percent, 2),
            "distance_flown_m": round(self.distance_flown_m, 1),
            "visited_nodes": list(self.visited_nodes),
            "is_loitering": self.is_loitering,
            "is_landed": self.is_landed,
            "is_feasible": self.is_feasible,
        }


@dataclass
class FlightLegTimeline:
    """Temporal and kinematic boundaries for a single flight leg."""
    leg_idx: int
    from_wp: Waypoint
    to_wp: Waypoint
    start_time_s: float
    transit_end_time_s: float
    loiter_end_time_s: float
    transit_duration_s: float
    loiter_duration_s: float
    distance_m: float
    ground_speed_mps: float
    heading_deg: float
    transit_power_w: float
    loiter_power_w: float
    transit_energy_joules: float
    loiter_energy_joules: float
    start_energy_joules: float
    start_distance_m: float


class KinematicFlightSimulator:
    """High-fidelity continuous kinematic flight simulator for a UAV SAR tour."""

    def __init__(
        self,
        drone_model: DronePhysicsModel,
        wind_field: WindField,
        waypoints: List[Waypoint],
        path: List[int],
    ) -> None:
        self.drone_model = drone_model
        self.wind_field = wind_field
        self.waypoints = waypoints
        self.path = path
        self.wp_map = {wp.id: wp for wp in waypoints}
        self.legs: List[FlightLegTimeline] = []
        self._build_timeline()

    def _build_timeline(self) -> None:
        """Construct chronologically continuous flight and loiter intervals."""
        current_time = 0.0
        current_energy = 0.0
        current_distance = 0.0

        p_transit = self.drone_model.power_consumption(self.drone_model.nominal_airspeed_mps)
        p_loiter = self.drone_model.hover_power_watts

        for i in range(len(self.path) - 1):
            u_idx = self.path[i]
            v_idx = self.path[i + 1]
            u_wp = self.waypoints[u_idx]
            v_wp = self.waypoints[v_idx]

            disp = v_wp.position_vec - u_wp.position_vec
            dist = float(np.linalg.norm(disp))

            if dist > 1e-4:
                try:
                    vg, _, _ = self.drone_model.solve_wind_triangle(disp, self.wind_field)
                except ValueError:
                    vg = max(0.5, self.drone_model.nominal_airspeed_mps - self.wind_field.speed_mps)
                transit_time = dist / max(0.5, vg)
                # Heading in degrees (0 = East, 90 = North)
                heading = math.degrees(math.atan2(disp[1], disp[0])) % 360.0
            else:
                vg = self.drone_model.nominal_airspeed_mps
                transit_time = 0.0
                heading = 0.0

            # Loiter only at survivor nodes, not upon final depot return
            is_final_return = (v_idx == 0 and i == len(self.path) - 2)
            loiter_time = 0.0 if is_final_return else v_wp.loiter_time_s

            transit_energy = p_transit * transit_time
            loiter_energy = p_loiter * loiter_time

            leg = FlightLegTimeline(
                leg_idx=i + 1,
                from_wp=u_wp,
                to_wp=v_wp,
                start_time_s=current_time,
                transit_end_time_s=current_time + transit_time,
                loiter_end_time_s=current_time + transit_time + loiter_time,
                transit_duration_s=transit_time,
                loiter_duration_s=loiter_time,
                distance_m=dist,
                ground_speed_mps=vg,
                heading_deg=heading,
                transit_power_w=p_transit,
                loiter_power_w=p_loiter,
                transit_energy_joules=transit_energy,
                loiter_energy_joules=loiter_energy,
                start_energy_joules=current_energy,
                start_distance_m=current_distance,
            )
            self.legs.append(leg)

            current_time += (transit_time + loiter_time)
            current_energy += (transit_energy + loiter_energy)
            current_distance += dist

        self.total_mission_time_s = current_time
        self.total_energy_joules = current_energy
        self.total_distance_m = current_distance

    def sample_state(self, t: float) -> DroneFlightSnapshot:
        """Sample instantaneous drone physical telemetry at exact mission time t."""
        if not self.legs:
            depot = self.waypoints[0]
            return DroneFlightSnapshot(
                time_s=t, x=depot.x, y=depot.y, lat=depot.lat, lon=depot.lon,
                heading_deg=0.0, ground_speed_mps=0.0, airspeed_mps=0.0,
                current_leg_idx=0, from_node_id=0, to_node_id=0, target_label="DEPOT",
                status="LANDED", power_watts=0.0, energy_consumed_wh=0.0,
                energy_consumed_joules=0.0, soc_percent=100.0, distance_flown_m=0.0,
                visited_nodes=[0], is_loitering=False, is_landed=True, is_feasible=True,
            )

        # Before takeoff
        if t <= 0.0:
            first_leg = self.legs[0]
            depot = first_leg.from_wp
            return DroneFlightSnapshot(
                time_s=0.0, x=depot.x, y=depot.y, lat=depot.lat, lon=depot.lon,
                heading_deg=first_leg.heading_deg, ground_speed_mps=0.0,
                airspeed_mps=0.0, current_leg_idx=1, from_node_id=depot.id,
                to_node_id=first_leg.to_wp.id, target_label=first_leg.to_wp.label,
                status="TAKEOFF_PREP", power_watts=self.drone_model.hover_power_watts,
                energy_consumed_wh=0.0, energy_consumed_joules=0.0,
                soc_percent=100.0, distance_flown_m=0.0, visited_nodes=[0],
                is_loitering=False, is_landed=False, is_feasible=True,
            )

        # After touchdown / Return to Launch Complete
        if t >= self.total_mission_time_s:
            final_leg = self.legs[-1]
            depot = final_leg.to_wp
            total_wh = self.total_energy_joules / 3600.0
            soc = self.drone_model.state_of_charge_percent(self.total_energy_joules)
            feasible = self.drone_model.is_mission_feasible(self.total_energy_joules)
            return DroneFlightSnapshot(
                time_s=t, x=depot.x, y=depot.y, lat=depot.lat, lon=depot.lon,
                heading_deg=final_leg.heading_deg, ground_speed_mps=0.0, airspeed_mps=0.0,
                current_leg_idx=len(self.legs), from_node_id=final_leg.from_wp.id,
                to_node_id=depot.id, target_label="BASE_DEPOT", status="TOUCHDOWN_RTL_COMPLETE",
                power_watts=0.0, energy_consumed_wh=total_wh,
                energy_consumed_joules=self.total_energy_joules, soc_percent=soc,
                distance_flown_m=self.total_distance_m,
                visited_nodes=list(set(self.path)), is_loitering=False,
                is_landed=True, is_feasible=feasible,
            )

        # Locate active leg interval
        for leg in self.legs:
            if leg.start_time_s <= t <= leg.loiter_end_time_s:
                # Track visited nodes up to this point
                visited = []
                for prev_l in self.legs:
                    if prev_l.leg_idx < leg.leg_idx:
                        visited.append(prev_l.from_wp.id)
                        visited.append(prev_l.to_wp.id)
                visited.append(leg.from_wp.id)
                visited = list(dict.fromkeys(visited))

                if t <= leg.transit_end_time_s:
                    # In Transit across terrain vector
                    delta_t = t - leg.start_time_s
                    frac = delta_t / max(1e-4, leg.transit_duration_s)
                    x = leg.from_wp.x + frac * (leg.to_wp.x - leg.from_wp.x)
                    y = leg.from_wp.y + frac * (leg.to_wp.y - leg.from_wp.y)
                    lat = leg.from_wp.lat + frac * (leg.to_wp.lat - leg.from_wp.lat)
                    lon = leg.from_wp.lon + frac * (leg.to_wp.lon - leg.from_wp.lon)

                    power = leg.transit_power_w
                    energy_j = leg.start_energy_joules + (power * delta_t)
                    dist_m = leg.start_distance_m + (leg.ground_speed_mps * delta_t)

                    is_rtl = (leg.to_wp.id == 0 and leg.leg_idx == len(self.legs))
                    status = "RTL_APPROACH_CAMP" if is_rtl else f"TRANSIT_TO_{leg.to_wp.label}"

                    soc = self.drone_model.state_of_charge_percent(energy_j)
                    feasible = (soc >= self.drone_model.reserve_margin_ratio * 100.0)

                    return DroneFlightSnapshot(
                        time_s=t, x=x, y=y, lat=lat, lon=lon,
                        heading_deg=leg.heading_deg, ground_speed_mps=leg.ground_speed_mps,
                        airspeed_mps=self.drone_model.nominal_airspeed_mps,
                        current_leg_idx=leg.leg_idx, from_node_id=leg.from_wp.id,
                        to_node_id=leg.to_wp.id, target_label=leg.to_wp.label,
                        status=status, power_watts=power,
                        energy_consumed_wh=energy_j / 3600.0,
                        energy_consumed_joules=energy_j, soc_percent=soc,
                        distance_flown_m=dist_m, visited_nodes=visited,
                        is_loitering=False, is_landed=False, is_feasible=feasible,
                    )
                else:
                    # Loitering / Scanning Survivor Beacon
                    delta_t = t - leg.transit_end_time_s
                    x = leg.to_wp.x
                    y = leg.to_wp.y
                    lat = leg.to_wp.lat
                    lon = leg.to_wp.lon

                    visited.append(leg.to_wp.id)
                    visited = list(dict.fromkeys(visited))

                    power = leg.loiter_power_w
                    energy_j = leg.start_energy_joules + leg.transit_energy_joules + (power * delta_t)
                    dist_m = leg.start_distance_m + leg.distance_m

                    soc = self.drone_model.state_of_charge_percent(energy_j)
                    feasible = (soc >= self.drone_model.reserve_margin_ratio * 100.0)

                    return DroneFlightSnapshot(
                        time_s=t, x=x, y=y, lat=lat, lon=lon,
                        heading_deg=leg.heading_deg, ground_speed_mps=0.0,
                        airspeed_mps=0.0, current_leg_idx=leg.leg_idx,
                        from_node_id=leg.from_wp.id, to_node_id=leg.to_wp.id,
                        target_label=leg.to_wp.label,
                        status=f"SURVEYING_{leg.to_wp.label}", power_watts=power,
                        energy_consumed_wh=energy_j / 3600.0,
                        energy_consumed_joules=energy_j, soc_percent=soc,
                        distance_flown_m=dist_m, visited_nodes=visited,
                        is_loitering=True, is_landed=False, is_feasible=feasible,
                    )

        # Fallback to last state
        return self.sample_state(self.total_mission_time_s)


class DualDroneRaceEngine:
    """Manages synchronized comparative live simulation between Proposed DP and Traditional Baseline."""

    def __init__(
        self,
        drone_model: DronePhysicsModel,
        wind_field: WindField,
        waypoints: List[Waypoint],
        dp_result: OptimizationResult,
        traditional_result: OptimizationResult,
    ) -> None:
        self.drone_model = drone_model
        self.wind_field = wind_field
        self.waypoints = waypoints
        self.dp_result = dp_result
        self.traditional_result = traditional_result

        self.sim_dp = KinematicFlightSimulator(
            drone_model, wind_field, waypoints, dp_result.optimal_path
        )
        self.sim_trad = KinematicFlightSimulator(
            drone_model, wind_field, waypoints, traditional_result.optimal_path
        )

        self.max_duration_s = max(self.sim_dp.total_mission_time_s, self.sim_trad.total_mission_time_s)

    def generate_timeline_frames(self, num_frames: int = 120) -> List[Dict[str, Any]]:
        """Generate synchronized temporal telemetry frames for 60 FPS Canvas playback."""
        time_steps = np.linspace(0.0, self.max_duration_s, num_frames)
        frames = []

        for t in time_steps:
            dp_snap = self.sim_dp.sample_state(float(t))
            trad_snap = self.sim_trad.sample_state(float(t))

            energy_delta_wh = trad_snap.energy_consumed_wh - dp_snap.energy_consumed_wh
            savings_pct = (
                (energy_delta_wh / trad_snap.energy_consumed_wh) * 100.0
                if trad_snap.energy_consumed_wh > 0.1
                else 0.0
            )

            frames.append({
                "t": round(float(t), 2),
                "dp": dp_snap.to_dict(),
                "trad": trad_snap.to_dict(),
                "energy_delta_wh": round(energy_delta_wh, 2),
                "savings_pct": round(savings_pct, 1),
            })

        return frames

    def get_summary_comparison(self) -> Dict[str, Any]:
        """Produce final comparative debriefing scorecard."""
        dp_final_wh = self.sim_dp.total_energy_joules / 3600.0
        trad_final_wh = self.sim_trad.total_energy_joules / 3600.0
        savings_wh = max(0.0, trad_final_wh - dp_final_wh)
        savings_pct = (savings_wh / trad_final_wh) * 100.0 if trad_final_wh > 0 else 0.0

        time_diff_s = self.sim_trad.total_mission_time_s - self.sim_dp.total_mission_time_s

        dp_soc = self.drone_model.state_of_charge_percent(self.sim_dp.total_energy_joules)
        trad_soc = self.drone_model.state_of_charge_percent(self.sim_trad.total_energy_joules)

        dp_feasible = self.drone_model.is_mission_feasible(self.sim_dp.total_energy_joules)
        trad_feasible = self.drone_model.is_mission_feasible(self.sim_trad.total_energy_joules)

        return {
            "dp_name": self.dp_result.algorithm_name,
            "trad_name": self.traditional_result.algorithm_name,
            "dp_path": self.dp_result.optimal_path,
            "trad_path": self.traditional_result.optimal_path,
            "dp_energy_wh": round(dp_final_wh, 2),
            "trad_energy_wh": round(trad_final_wh, 2),
            "energy_savings_wh": round(savings_wh, 2),
            "energy_savings_pct": round(savings_pct, 1),
            "dp_time_min": round(self.sim_dp.total_mission_time_s / 60.0, 2),
            "trad_time_min": round(self.sim_trad.total_mission_time_s / 60.0, 2),
            "time_diff_s": round(time_diff_s, 1),
            "dp_distance_km": round(self.sim_dp.total_distance_m / 1000.0, 2),
            "trad_distance_km": round(self.sim_trad.total_distance_m / 1000.0, 2),
            "dp_soc": round(dp_soc, 1),
            "trad_soc": round(trad_soc, 1),
            "dp_feasible": dp_feasible,
            "trad_feasible": trad_feasible,
            "reserve_threshold_pct": round(self.drone_model.reserve_margin_ratio * 100.0, 1),
            "max_duration_s": round(self.max_duration_s, 1),
        }
