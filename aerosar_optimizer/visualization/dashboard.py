"""
visualization/dashboard.py - Interactive Streamlit Ground Control Station Dashboard & Real-Time Race Simulator.

Production-grade real-time autonomous UAV search-and-rescue mission planner:
1. Dynamic parameter tuning (survivor beacons count n, wind vectors, drone aerodynamics, battery limits).
2. Live 60 FPS Tactical Radar HUD: Real-time dual-drone flight race comparing Proposed Held-Karp
   Bitmask DP vs. Traditional Baselines (Greedy NN, Euclidean Wind-Blind, Brute Force).
3. Real-time battery depletion, ground speed kinematics, wind drift crab angles, and RTL landing flare.
4. Interactive Plotly telemetry curves: Battery SoC (%) vs Flight Time and Energy vs Distance.
5. Exportable QGroundControl / MAVLink .waypoints mission file generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct streamlit execution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import io
import math
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from core.physics import DronePhysicsModel, WindField, Waypoint
from core.cost_matrix import CostMatrixCalculator
from core.bitmask_dp import BitmaskDPSolver, OptimizationResult
import importlib
try:
    import core.brute_force
    importlib.reload(core.brute_force)
    from core.brute_force import (
        GreedyNearestNeighborSolver,
        BruteForceSolver,
        EuclideanNearestNeighborSolver,
    )
except (ImportError, AttributeError):
    import core.brute_force
    from core.brute_force import GreedyNearestNeighborSolver, BruteForceSolver
    if hasattr(core.brute_force, "EuclideanNearestNeighborSolver"):
        EuclideanNearestNeighborSolver = core.brute_force.EuclideanNearestNeighborSolver
    else:
        # Fallback alias to Greedy solver if cache cannot be purged
        EuclideanNearestNeighborSolver = GreedyNearestNeighborSolver

from core.telemetry import MAVLinkMissionGenerator
from simulation.synthetic_data import SyntheticScenarioGenerator
import simulation.live_simulation
importlib.reload(simulation.live_simulation)
from simulation.live_simulation import DualDroneRaceEngine

import visualization.live_radar_component
importlib.reload(visualization.live_radar_component)
from visualization.plotter import MissionPlotter
from visualization.live_radar_component import render_live_radar_html


# Page configuration
st.set_page_config(
    page_title="AeroSAR-Optimizer | Autonomous UAV Route Planner & Real-Time Simulator",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber-Physical Aerospace Dark Theme Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #111827 0%, #090D16 100%);
        color: #E2E8F0;
    }

    /* KPI Card Containers */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }

    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .kpi-sub {
        font-size: 0.78rem;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Status Badges */
    .badge-feasible {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }

    .badge-infeasible {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }

    .hud-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .hud-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 24px;
    }

    .race-banner {
        background: linear-gradient(90deg, rgba(6, 182, 212, 0.12) 0%, rgba(249, 115, 22, 0.12) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # Header HUD
    st.markdown('<div class="hud-title">AEROSAR-OPTIMIZER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud-subtitle">'
        'Cyber-Physical UAV Search-and-Rescue Route Optimization Engine | '
        'Asymmetric Bitmask Dynamic Programming & Real-Time Flight Simulation'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Tactical Controls
    with st.sidebar:
        st.header("⚙️ Tactical Mission Parameters")

        scenario_preset = st.selectbox(
            "SAR Scenario Environment",
            options=["ALPINE_AVALANCHE", "WILDERNESS_DISPERSED", "COASTAL_MARITIME"],
            index=0,
            help="Terrain distribution pattern of survivor locator beacons",
        )

        num_survivors = st.slider(
            "Survivor Rescue Nodes (n)",
            min_value=4,
            max_value=15,
            value=10,
            step=1,
            help="Number of survivor targets to triage and survey (excluding Base Depot)",
        )

        baseline_algo_options = [
            "Greedy Nearest-Neighbor (Energy)",
            "Traditional Distance-First (Wind-Blind)",
        ]
        if num_survivors <= 8:
            baseline_algo_options.append("Brute-Force Permutations O(n!)")

        baseline_algo_choice = st.selectbox(
            "Traditional Algorithm Baseline",
            options=baseline_algo_options,
            index=0,
            help="Traditional benchmark algorithm to pit against Proposed Bitmask DP in real-time simulation",
        )

        random_seed = st.number_input(
            "Synthetic Cluster Seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            help="Change to generate different survivor spatial distributions",
        )

        st.markdown("---")
        st.subheader("💨 Atmospheric Wind Field")

        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=16.0,
            value=6.8,
            step=0.2,
            help="Horizontal atmospheric wind magnitude (critical for aerodynamic drift)",
        )

        wind_direction = st.slider(
            "Wind Direction (Degrees from East)",
            min_value=0.0,
            max_value=360.0,
            value=55.0,
            step=5.0,
            help="Direction towards which the wind vector points (0° = East, 90° = North)",
        )

        st.markdown("---")
        st.subheader("🔋 Multirotor UAV & Battery")

        airspeed = st.slider(
            "Nominal True Airspeed V_air (m/s)",
            min_value=10.0,
            max_value=24.0,
            value=15.0,
            step=0.5,
        )

        battery_capacity = st.slider(
            "Battery Pack Capacity (Wh)",
            min_value=150.0,
            max_value=600.0,
            value=350.0,
            step=10.0,
        )

        reserve_margin = st.slider(
            "Safety Reserve Margin (%)",
            min_value=10,
            max_value=35,
            value=20,
            step=1,
            help="Mandatory buffer reserved for landing and unexpected gusts",
        ) / 100.0

        loiter_time = st.slider(
            "Sensor Scan Time per Target (s)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Hover duration at each survivor beacon for thermal/optical imaging",
        )

        st.markdown("---")
        st.caption("AeroSAR-Optimizer v2.4.0-Production | DAA Systems")

    # Instantiate Drone & Physics Model
    drone = DronePhysicsModel(
        nominal_airspeed_mps=airspeed,
        hover_power_watts=310.0,
        drag_coefficient=0.048,
        battery_capacity_wh=battery_capacity,
        reserve_margin_ratio=reserve_margin,
    )

    wind = WindField(speed_mps=wind_speed, direction_deg=wind_direction)

    # Generate Synthetic Scenario
    generator = SyntheticScenarioGenerator()
    scenario = generator.generate(
        num_survivors=num_survivors,
        wind_speed_mps=wind.speed_mps,
        wind_direction_deg=wind.direction_deg,
        scenario_preset=scenario_preset,
        seed=int(random_seed),
    )

    waypoints = scenario.waypoints
    for wp in waypoints[1:]:
        wp.loiter_time_s = loiter_time

    # Compute Directed Asymmetric Cost Matrix
    calculator = CostMatrixCalculator(drone, wind)
    cost_res = calculator.compute(waypoints)

    # Execute Proposed Bitmask DP Solver
    dp_solver = BitmaskDPSolver(drone)
    dp_result = dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)

    # Execute Traditional Baseline Solver
    if "Distance-First" in baseline_algo_choice:
        baseline_solver = EuclideanNearestNeighborSolver(drone)
        trad_result = baseline_solver.solve(cost_res, waypoints)
    elif "Brute-Force" in baseline_algo_choice and num_survivors <= 8:
        baseline_solver = BruteForceSolver(drone)
        trad_result = baseline_solver.solve(cost_res, waypoints)
    else:
        baseline_solver = GreedyNearestNeighborSolver(drone)
        trad_result = baseline_solver.solve(cost_res, waypoints)

    # Build Live Race Simulation Engine
    race_engine = DualDroneRaceEngine(
        drone_model=drone,
        wind_field=wind,
        waypoints=waypoints,
        dp_result=dp_result,
        traditional_result=trad_result,
    )
    frames = race_engine.generate_timeline_frames(num_frames=150)
    sim_summary = race_engine.get_summary_comparison()

    # Top KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        feas_html = (
            '<span class="badge-feasible">✓ FEASIBLE</span>'
            if dp_result.is_feasible
            else '<span class="badge-infeasible">⚠ INFEASIBLE</span>'
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Mission Envelope</div>
                <div style="margin-top: 8px; margin-bottom: 8px;">{feas_html}</div>
                <div class="kpi-sub">Limit: {drone.usable_energy_wh:.1f} Wh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">DP Optimal Energy</div>
                <div class="kpi-value">{dp_result.total_energy_wh:.1f} <span style="font-size:1rem;color:#94A3B8;">Wh</span></div>
                <div class="kpi-sub">{dp_result.total_energy_joules / 1000.0:.1f} kJ consumed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        savings_wh = max(0.0, trad_result.total_energy_wh - dp_result.total_energy_wh)
        savings_pct = (
            (savings_wh / trad_result.total_energy_wh) * 100.0
            if trad_result.total_energy_wh > 0
            else 0.0
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">DP vs Baseline Delta</div>
                <div class="kpi-value" style="color:#10B981;">-{savings_pct:.1f}%</div>
                <div class="kpi-sub">{savings_wh:.1f} Wh saved over {trad_result.algorithm_name.split()[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        flight_min = dp_result.total_flight_time_s / 60.0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Flight Sortie Time</div>
                <div class="kpi-value">{flight_min:.1f} <span style="font-size:1rem;color:#94A3B8;">min</span></div>
                <div class="kpi-sub">{dp_result.total_flight_time_s:.0f}s total mission</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Asymmetry Index α</div>
                <div class="kpi-value" style="color:#38BDF8;">{cost_res.asymmetry_index:.3f}</div>
                <div class="kpi-sub">Peak Skew: {cost_res.max_asymmetry_ratio * 100.0:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Battery State of Charge Bar
    soc_col1, soc_col2 = st.columns([3, 1])
    with soc_col1:
        soc_progress = max(0.0, min(1.0, dp_result.remaining_soc_percent / 100.0))
        st.progress(
            soc_progress,
            text=f"Battery Reserve Gauge: {dp_result.remaining_soc_percent:.1f}% Remaining upon Base Landing (Reserve Margin Buffer: {drone.reserve_margin_ratio * 100.0:.0f}%)",
        )
    with soc_col2:
        st.metric(
            label="DP Solve Runtime",
            value=f"{dp_result.execution_time_s * 1000.0:.2f} ms",
            delta=f"{dp_result.num_evaluations:,} states",
        )

    # Main Navigation Tabs
    tab_sim, tab_traj, tab_bench, tab_mavlink = st.tabs([
        "⚡ Real-Time Flight Simulation & Algorithmic Duel",
        "📍 Tactical Trajectory & Static Telemetry",
        "📊 Algorithmic Benchmarks & Complexity Scaling",
        "📡 MAVLink Mission Telemetry Export",
    ])

    # =========================================================================
    # TAB 1: LIVE REAL-TIME DUAL-DRONE FLIGHT SIMULATION
    # =========================================================================
    with tab_sim:
        st.markdown(
            f"""
            <div class="race-banner">
                <div>
                    <strong style="color:#38BDF8; font-size:1.1rem;">🚁 Real-Time UAV Sortie Duel:</strong>
                    <span style="color:#E2E8F0;"> Proposed <strong>Held-Karp Bitmask DP</strong> (Cyan) vs. <strong>{trad_result.algorithm_name}</strong> (Amber)</span>
                </div>
                <div>
                    <span style="font-family:'JetBrains Mono',monospace; color:#10B981; font-weight:700;">
                        Net Energy Savings: {sim_summary['energy_savings_wh']} Wh ({sim_summary['energy_savings_pct']}%)
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 60 FPS HTML5 Canvas Tactical Radar HUD Component
        radar_html = render_live_radar_html(
            waypoints=waypoints,
            wind_field=wind,
            drone_model=drone,
            frames=frames,
            summary=sim_summary,
        )
        components.html(radar_html, height=660, scrolling=False)

        st.markdown("---")

        # Real-Time Telemetry Comparison Curves (Plotly)
        st.subheader("📈 Real-Time Kinematics & Energy Depletion Telemetry")

        chart_col1, chart_col2 = st.columns(2)

        # Build timeline series for Plotly
        t_series = [f["t"] for f in frames]
        dp_soc_series = [f["dp"]["soc_percent"] for f in frames]
        trad_soc_series = [f["trad"]["soc_percent"] for f in frames]
        dp_e_series = [f["dp"]["energy_consumed_wh"] for f in frames]
        trad_e_series = [f["trad"]["energy_consumed_wh"] for f in frames]
        reserve_line = [drone.reserve_margin_ratio * 100.0] * len(frames)

        with chart_col1:
            fig_soc = go.Figure()
            fig_soc.add_trace(go.Scatter(
                x=t_series,
                y=dp_soc_series,
                mode="lines",
                name="AeroSAR Bitmask DP",
                line=dict(color="#06B6D4", width=3),
            ))
            fig_soc.add_trace(go.Scatter(
                x=t_series,
                y=trad_soc_series,
                mode="lines",
                name=trad_result.algorithm_name,
                line=dict(color="#F97316", width=2.5, dash="dash"),
            ))
            fig_soc.add_trace(go.Scatter(
                x=t_series,
                y=reserve_line,
                mode="lines",
                name="Safety Reserve Limit (20%)",
                line=dict(color="#EF4444", width=1.5, dash="dot"),
            ))
            fig_soc.update_layout(
                title="Battery State-of-Charge (SoC %) vs Elapsed Flight Time",
                xaxis_title="Mission Flight Elapsed Time (seconds)",
                yaxis_title="Remaining Battery SoC (%)",
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#0B1120",
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_soc, use_container_width=True)

        with chart_col2:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=t_series,
                y=dp_e_series,
                mode="lines",
                name="AeroSAR DP Energy",
                line=dict(color="#10B981", width=3),
            ))
            fig_e.add_trace(go.Scatter(
                x=t_series,
                y=trad_e_series,
                mode="lines",
                name=f"{trad_result.algorithm_name} Energy",
                line=dict(color="#F59E0B", width=2.5, dash="dash"),
            ))
            fig_e.update_layout(
                title="Cumulative Energy Consumed (Wh) vs Sortie Duration",
                xaxis_title="Mission Flight Elapsed Time (seconds)",
                yaxis_title="Cumulative Energy Consumed (Wh)",
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#0B1120",
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_e, use_container_width=True)

        # Comparative Sortie Debriefing Scorecard
        st.subheader("🏁 Head-to-Head Mission Debriefing Scorecard")
        deb_col1, deb_col2, deb_col3, deb_col4 = st.columns(4)

        with deb_col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Energy Draw</div>
                    <div class="kpi-value" style="font-size:1.45rem;">
                        <span style="color:#06B6D4;">{sim_summary['dp_energy_wh']} Wh</span> vs <span style="color:#F97316;">{sim_summary['trad_energy_wh']} Wh</span>
                    </div>
                    <div class="kpi-sub" style="color:#10B981;">Saved {sim_summary['energy_savings_wh']} Wh ({sim_summary['energy_savings_pct']}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with deb_col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Sortie Duration</div>
                    <div class="kpi-value" style="font-size:1.45rem;">
                        <span style="color:#06B6D4;">{sim_summary['dp_time_min']} min</span> vs <span style="color:#F97316;">{sim_summary['trad_time_min']} min</span>
                    </div>
                    <div class="kpi-sub">Delta: {abs(sim_summary['time_diff_s']):.1f}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with deb_col3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Touchdown Battery SoC</div>
                    <div class="kpi-value" style="font-size:1.45rem;">
                        <span style="color:#06B6D4;">{sim_summary['dp_soc']}%</span> vs <span style="color:#F97316;">{sim_summary['trad_soc']}%</span>
                    </div>
                    <div class="kpi-sub">Threshold: {sim_summary['reserve_threshold_pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with deb_col4:
            dp_pass = "✓ PASS (SAFE)" if sim_summary["dp_feasible"] else "⚠ DEFICIT"
            trad_pass = "✓ PASS (SAFE)" if sim_summary["trad_feasible"] else "⚠ INFEASIBLE"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Reserve Compliance</div>
                    <div class="kpi-value" style="font-size:1.25rem;">
                        <span style="color:#06B6D4;">{dp_pass}</span> / <span style="color:#F97316;">{trad_pass}</span>
                    </div>
                    <div class="kpi-sub">Base Camp Return Verified</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Detailed Order Comparison
        st.markdown(f"**Optimal Bitmask DP Traversal Order:** `{sim_summary['dp_path']}`")
        st.markdown(f"**Traditional Baseline Traversal Order:** `{sim_summary['trad_path']}`")

    # =========================================================================
    # TAB 2: STATIC TACTICAL TRAJECTORY & TELEMETRY
    # =========================================================================
    with tab_traj:
        st.subheader("Tactical Flight Trajectory Map")
        plotter = MissionPlotter(dark_mode=True)
        fig = plotter.plot_trajectories(
            waypoints=waypoints,
            dp_result=dp_result,
            greedy_result=trad_result,
            wind_field=wind,
            title=f"AeroSAR: Optimal Held-Karp Tour vs {trad_result.algorithm_name} (Wind: {wind.speed_mps} m/s @ {wind.direction_deg}°)",
            figsize=(12, 8),
        )
        st.pyplot(fig, use_container_width=True)

        if not dp_result.is_feasible:
            st.warning(
                f"⚠️ BATTERY WARNING: Tour energy ({dp_result.total_energy_wh:.1f} Wh) exceeds "
                f"usable capacity ({drone.usable_energy_wh:.1f} Wh). "
                f"Deficit: {dp_result.total_energy_wh - drone.usable_energy_wh:.1f} Wh."
            )
            # Show automated triage solution
            triage_res = dp_solver.solve(cost_res, waypoints, enforce_feasibility=True)
            if triage_res.triage_applied:
                st.info(f"🚑 **Automated Triage Contingency Route Activated:**\n{triage_res.triage_notes}")

        st.subheader("Step-by-Step Leg Kinematics Telemetry")
        if dp_result.leg_details:
            df_legs = pd.DataFrame(dp_result.leg_details)
            df_display = df_legs[[
                "leg_index",
                "from_label",
                "to_label",
                "distance_m",
                "ground_speed_mps",
                "duration_s",
                "energy_wh",
                "cumulative_energy_wh",
                "remaining_soc_pct",
            ]].copy()
            df_display.columns = [
                "Leg #",
                "Origin",
                "Target",
                "Distance (m)",
                "Ground Speed (m/s)",
                "Duration (s)",
                "Leg Energy (Wh)",
                "Cum. Energy (Wh)",
                "Remaining SoC (%)",
            ]
            st.dataframe(
                df_display.style.format({
                    "Distance (m)": "{:.1f}",
                    "Ground Speed (m/s)": "{:.2f}",
                    "Duration (s)": "{:.1f}",
                    "Leg Energy (Wh)": "{:.2f}",
                    "Cum. Energy (Wh)": "{:.2f}",
                    "Remaining SoC (%)": "{:.1f}%",
                }),
                use_container_width=True,
            )

    # =========================================================================
    # TAB 3: COMPLEXITY & BENCHMARKS
    # =========================================================================
    with tab_bench:
        st.subheader("Algorithmic Complexity & Scalability Evaluation")
        st.markdown(
            """
            Comparison of exact **Held-Karp Bitmask DP ($O(n^2 2^n)$)** against
            **Permutation Brute Force ($O(n!)$)** and **Greedy Nearest Neighbor ($O(n^2)$)**.
            """
        )

        n_range = [4, 6, 8, 10, 12, 14]
        bf_theoretical = [math.factorial(n) * 2.5e-7 for n in n_range]
        dp_theoretical = [(n**2 * (2**n)) * 3e-8 for n in n_range]

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Theoretical Time Complexity Scaling")
            fig_comp, ax_comp = plt.subplots(figsize=(6, 4), facecolor="#0B0F19")
            ax_comp.set_facecolor("#0B0F19")
            ax_comp.plot(n_range, bf_theoretical, "r-o", label="Brute Force O(n!)")
            ax_comp.plot(n_range, dp_theoretical, "c-s", label="Bitmask DP O(n² 2ⁿ)")
            ax_comp.set_yscale("log")
            ax_comp.set_xlabel("Number of Survivor Targets (n)", color="#E2E8F0")
            ax_comp.set_ylabel("Execution Time (seconds, log scale)", color="#E2E8F0")
            ax_comp.grid(True, linestyle=":", color="#374151")
            ax_comp.tick_params(colors="#E2E8F0")
            ax_comp.legend(facecolor="#1F2937", labelcolor="#E2E8F0")
            for spine in ax_comp.spines.values():
                spine.set_color("#374151")
            st.pyplot(fig_comp, use_container_width=True)

        with col_b2:
            st.markdown("#### Dynamic Programming Memory Footprint")
            mem_kb = [((1 << (n + 1)) * (n + 1) * 10) / 1024.0 for n in n_range]
            fig_mem, ax_mem = plt.subplots(figsize=(6, 4), facecolor="#0B0F19")
            ax_mem.set_facecolor("#0B0F19")
            ax_mem.bar([str(n) for n in n_range], mem_kb, color="#818CF8", alpha=0.85)
            ax_mem.set_xlabel("Survivor Targets (n)", color="#E2E8F0")
            ax_mem.set_ylabel("Memory Footprint (KB)", color="#E2E8F0")
            ax_mem.grid(True, linestyle=":", color="#374151")
            ax_mem.tick_params(colors="#E2E8F0")
            for spine in ax_mem.spines.values():
                spine.set_color("#374151")
            st.pyplot(fig_mem, use_container_width=True)

    # =========================================================================
    # TAB 4: MAVLINK MISSION TELEMETRY EXPORT
    # =========================================================================
    with tab_mavlink:
        st.subheader("MAVLink / QGroundControl Waypoint Mission Generator")
        st.markdown(
            "Exports the computed trajectory into an industry-standard **QGroundControl WPL 110 (.waypoints)** file. "
            "Load directly into QGroundControl, Mission Planner, or PX4 Autopilot to execute the autonomous SAR sortie."
        )

        telemetry_gen = MAVLinkMissionGenerator(
            survey_alt_agl_m=50.0,
            loiter_time_s=loiter_time,
            acceptance_radius_m=3.0,
        )

        waypoints_text = telemetry_gen.generate_waypoints_text(
            optimal_path=dp_result.optimal_path,
            waypoints=waypoints,
            include_rtl=True,
        )

        st.download_button(
            label="📥 Download QGroundControl (.waypoints) Mission File",
            data=waypoints_text,
            file_name=f"aerosar_{scenario_preset.lower()}_n{num_survivors}.waypoints",
            mime="text/plain",
            help="Click to download the mission file for direct upload to QGroundControl GCS",
        )

        st.text_area(
            "Mission File Preview (QGC WPL 110)",
            value=waypoints_text,
            height=320,
            help="Tabs separate index, frame, command, parameters, latitude, longitude, and altitude.",
        )


if __name__ == "__main__":
    main()
