"""
visualization/dashboard.py - Interactive Streamlit Ground Control Station Dashboard.

Production-grade real-time autonomous UAV search-and-rescue mission planner:
1. Dynamic parameter tuning (survivors, wind vectors, battery capacity, reserve margin).
2. Cyber-physical kinematics and asymmetric cost matrix telemetry.
3. Held-Karp Bitmask DP vs. Greedy Trajectory visualization with aerodynamic quivers.
4. Battery depletion gauges, triage contingency advisory, and algorithmic scaling curves.
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
import numpy as np
import matplotlib.pyplot as plt

from core.physics import DronePhysicsModel, WindField, Waypoint
from core.cost_matrix import CostMatrixCalculator
from core.bitmask_dp import BitmaskDPSolver, OptimizationResult
from core.brute_force import GreedyNearestNeighborSolver, BruteForceSolver
from core.telemetry import MAVLinkMissionGenerator
from simulation.synthetic_data import SyntheticScenarioGenerator
from visualization.plotter import MissionPlotter


# Page configuration
st.set_page_config(
    page_title="AeroSAR-Optimizer | Autonomous UAV Route Planner",
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
        'Asymmetric Bitmask Dynamic Programming & Kinematic Wind Drift Simulation'
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
            "Survivor Beacon Count (n)",
            min_value=4,
            max_value=15,
            value=10,
            step=1,
            help="Number of survivor targets to triage and survey (excluding Base Depot)",
        )

        random_seed = st.number_input(
            "Synthetic Generator Seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
        )

        st.markdown("---")
        st.subheader("💨 Atmospheric Wind Field")

        wind_speed = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=16.0,
            value=6.8,
            step=0.2,
            help="Horizontal atmospheric wind magnitude",
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
            max_value=22.0,
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

    # Compute Directed Asymmetric Cost Matrix
    calculator = CostMatrixCalculator(drone, wind)
    cost_res = calculator.compute(waypoints)

    # Execute Solvers
    dp_solver = BitmaskDPSolver(drone)
    dp_result = dp_solver.solve(cost_res, waypoints, enforce_feasibility=False)

    greedy_solver = GreedyNearestNeighborSolver(drone)
    greedy_result = greedy_solver.solve(cost_res, waypoints)

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
        savings_wh = max(0.0, greedy_result.total_energy_wh - dp_result.total_energy_wh)
        savings_pct = (savings_wh / greedy_result.total_energy_wh) * 100.0 if greedy_result.total_energy_wh > 0 else 0.0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">DP vs Greedy Delta</div>
                <div class="kpi-value" style="color:#10B981;">-{savings_pct:.1f}%</div>
                <div class="kpi-sub">{savings_wh:.1f} Wh saved over Greedy</div>
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
                <div class="kpi-sub">{dp_result.total_flight_time_s:.0f} seconds duration</div>
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
                <div class="kpi-sub">Max Leg Skew: {cost_res.max_asymmetry_ratio * 100.0:.0f}%</div>
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
    tab1, tab2, tab3 = st.tabs([
        "📍 Tactical Trajectory & Telemetry",
        "📊 Algorithmic Benchmarks & Scaling",
        "📡 MAVLink Mission Telemetry Export",
    ])

    # TAB 1: Trajectory & Leg Breakdown
    with tab1:
        st.subheader("Tactical Flight Trajectory Map")
        plotter = MissionPlotter(dark_mode=True)
        fig = plotter.plot_trajectories(
            waypoints=waypoints,
            dp_result=dp_result,
            greedy_result=greedy_result,
            wind_field=wind,
            title=f"AeroSAR: Optimal Held-Karp Tour vs Greedy (Wind: {wind.speed_mps} m/s @ {wind.direction_deg}°)",
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
            import pandas as pd
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

    # TAB 2: Complexity & Benchmarks
    with tab2:
        st.subheader("Algorithmic Complexity & Scalability Evaluation")
        st.markdown(
            """
            Comparison of exact **Held-Karp Bitmask DP ($O(n^2 2^n)$)** against
            **Permutation Brute Force ($O(n!)$)** and **Greedy Nearest Neighbor ($O(n^2)$)**.
            """
        )

        n_range = [4, 6, 8, 10, 12, 14]
        bf_theoretical = [math.factorial(n) * 2.5e-7 for n in n_range]  # estimated seconds
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

    # TAB 3: Telemetry Export
    with tab3:
        st.subheader("MAVLink / QGroundControl Waypoint Mission Generator")
        st.markdown(
            "Exports the computed trajectory into an industry-standard **QGroundControl WPL 110 (.waypoints)** file. "
            "Load directly into QGroundControl, Mission Planner, or PX4 Autopilot to execute the autonomous SAR sortie."
        )

        telemetry_gen = MAVLinkMissionGenerator(
            survey_alt_agl_m=50.0,
            loiter_time_s=5.0,
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
