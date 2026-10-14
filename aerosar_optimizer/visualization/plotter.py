"""
visualization/plotter.py - Tactical UAV SAR Trajectory & Wind Vector Plotter.

Renders publication-grade 2D kinematic trajectory maps comparing:
1. Exact Bitmask DP Optimal Route vs. Greedy Nearest-Neighbor Heuristic.
2. Aerodynamic wind vector quiver fields illustrating drift and head/tail winds.
3. Base Depot, prioritized survivor beacons, and flight sequence telemetry HUD.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np

from core.physics import Waypoint, WindField
from core.bitmask_dp import OptimizationResult


class MissionPlotter:
    """Produces tactical aerospace visualizations for SAR UAV route optimization."""

    def __init__(self, dark_mode: bool = True) -> None:
        self.dark_mode = dark_mode
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure Matplotlib aesthetic parameters."""
        if self.dark_mode:
            self.bg_color = "#0B0F19"
            self.card_color = "#111827"
            self.grid_color = "#1F2937"
            self.text_color = "#F3F4F6"
            self.dp_color = "#00F0FF"       # Neon Cyan
            self.greedy_color = "#FF007F"   # Neon Pink / Magenta
            self.wind_color = "#4B5563"     # Subtle Grey-Blue
            self.depot_color = "#10B981"    # Neon Emerald Green
        else:
            self.bg_color = "#F9FAFB"
            self.card_color = "#FFFFFF"
            self.grid_color = "#E5E7EB"
            self.text_color = "#111827"
            self.dp_color = "#0284C7"       # Ocean Blue
            self.greedy_color = "#E11D48"   # Crimson
            self.wind_color = "#9CA3AF"
            self.depot_color = "#059669"

    def plot_trajectories(
        self,
        waypoints: List[Waypoint],
        dp_result: OptimizationResult,
        greedy_result: Optional[OptimizationResult] = None,
        wind_field: Optional[WindField] = None,
        title: str = "AeroSAR-Optimizer: Tactical Kinematic Trajectory",
        save_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (13, 11),
    ) -> plt.Figure:
        """Generate tactical 2D route comparison plot with aerodynamic quiver arrows."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)

        all_x = [wp.x for wp in waypoints]
        all_y = [wp.y for wp in waypoints]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pad_x = max(500.0, (max_x - min_x) * 0.25)
        # Extra top padding so HUD badge has dedicated clear space
        pad_y_bottom = max(400.0, (max_y - min_y) * 0.20)
        pad_y_top = max(750.0, (max_y - min_y) * 0.38)

        x_lims = (min_x - pad_x, max_x + pad_x)
        y_lims = (min_y - pad_y_bottom, max_y + pad_y_top)

        # 1. Background Wind Field Quiver
        if wind_field is not None and wind_field.speed_mps > 0.05:
            grid_res = 14
            gx = np.linspace(x_lims[0], x_lims[1], grid_res)
            gy = np.linspace(y_lims[0], y_lims[1], grid_res)
            grid_X, grid_Y = np.meshgrid(gx, gy)

            u_wind = np.full_like(grid_X, wind_field.vx)
            v_wind = np.full_like(grid_Y, wind_field.vy)

            ax.quiver(
                grid_X,
                grid_Y,
                u_wind,
                v_wind,
                color=self.wind_color,
                alpha=0.35,
                scale=wind_field.speed_mps * 18.0,
                width=0.0022,
                headwidth=3.5,
                headlength=4.5,
                zorder=1,
                label=f"Wind Vector ({wind_field.speed_mps:.1f} m/s @ {wind_field.direction_deg:.0f}°)",
            )

        # 2. Plot Greedy Route (Dashed Baseline)
        if greedy_result is not None and greedy_result.optimal_path:
            g_path = greedy_result.optimal_path
            g_xs = [waypoints[idx].x for idx in g_path]
            g_ys = [waypoints[idx].y for idx in g_path]

            ax.plot(
                g_xs,
                g_ys,
                color=self.greedy_color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.85,
                zorder=3,
                label=f"Greedy Nearest-Neighbor ({greedy_result.total_energy_wh:.1f} Wh)",
            )

        # 3. Plot Bitmask DP Route (Solid Glow Line + Directional Flight Segments)
        if dp_result.optimal_path:
            dp_path = dp_result.optimal_path
            dp_xs = [waypoints[idx].x for idx in dp_path]
            dp_ys = [waypoints[idx].y for idx in dp_path]

            # Primary optimal route line
            ax.plot(
                dp_xs,
                dp_ys,
                color=self.dp_color,
                linewidth=3.2,
                alpha=0.95,
                zorder=4,
                label=f"Bitmask DP Optimal ({dp_result.total_energy_wh:.1f} Wh)",
            )

            # Draw directional flow arrows along each leg
            for k in range(len(dp_path) - 1):
                u_idx = dp_path[k]
                v_idx = dp_path[k + 1]
                x1, y1 = waypoints[u_idx].x, waypoints[u_idx].y
                x2, y2 = waypoints[v_idx].x, waypoints[v_idx].y

                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy)
                if dist > 80.0:
                    mid_x = x1 + dx * 0.52
                    mid_y = y1 + dy * 0.52
                    ax.annotate(
                        "",
                        xy=(mid_x + dx * 0.05, mid_y + dy * 0.05),
                        xytext=(mid_x, mid_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=self.dp_color,
                            lw=2.5,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        zorder=5,
                    )
                    # Leg sequence number badge
                    seq_badge = f"{k + 1}"
                    ax.text(
                        mid_x - dy * 0.04,
                        mid_y + dx * 0.04,
                        seq_badge,
                        color="#000000",
                        fontsize=7,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="circle,pad=0.2", fc=self.dp_color, ec="none", alpha=0.9),
                        zorder=6,
                    )

        # 4. Plot Survivor Beacons
        survivor_wps = waypoints[1:]
        if survivor_wps:
            surv_xs = [wp.x for wp in survivor_wps]
            surv_ys = [wp.y for wp in survivor_wps]
            priorities = [wp.priority for wp in survivor_wps]

            norm = Normalize(vmin=1.0, vmax=10.0)
            scatter = ax.scatter(
                surv_xs,
                surv_ys,
                c=priorities,
                cmap="YlOrRd",
                norm=norm,
                s=240,
                edgecolors="#FFFFFF",
                linewidths=1.8,
                zorder=7,
            )

            # Labels for survivors
            for wp in survivor_wps:
                ax.annotate(
                    f"{wp.label}\n(P:{wp.priority:.1f})",
                    xy=(wp.x, wp.y),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                    color=self.text_color,
                    bbox=dict(boxstyle="round,pad=0.25", fc=self.card_color, ec="#374151", lw=0.8, alpha=0.85),
                    zorder=8,
                )

            # Colorbar for priority
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.032, pad=0.03)
            cbar.set_label("Survivor Triage Urgency Priority (1 - 10)", color=self.text_color, fontsize=9.5)
            cbar.ax.yaxis.set_tick_params(color=self.text_color)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=self.text_color, fontsize=8)

        # 5. Base Depot (Node 0)
        depot = waypoints[0]
        ax.scatter(
            [depot.x],
            [depot.y],
            color=self.depot_color,
            s=480,
            marker="P",
            edgecolors="#FFFFFF",
            linewidths=2.2,
            zorder=9,
            label="Base Camp / Launch Site (Depot)",
        )
        ax.annotate(
            f"BASE DEPOT\n(0, 0)",
            xy=(depot.x, depot.y),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=self.depot_color,
            bbox=dict(boxstyle="round,pad=0.3", fc=self.card_color, ec=self.depot_color, lw=1.2, alpha=0.9),
            zorder=10,
        )

        # 6. Telemetry HUD Overlay Card
        feasibility_status = "FEASIBLE" if dp_result.is_feasible else "INFEASIBLE (EXCEEDED)"
        feas_color = "#10B981" if dp_result.is_feasible else "#EF4444"

        savings_str = ""
        if greedy_result is not None and greedy_result.total_energy_wh > 0:
            diff_wh = greedy_result.total_energy_wh - dp_result.total_energy_wh
            diff_pct = (diff_wh / greedy_result.total_energy_wh) * 100.0
            savings_str = f"\nOptimal Savings: {diff_wh:.1f} Wh ({diff_pct:.1f}%)"

        hud_text = (
            f"AEROSAR FLIGHT TELEMETRY HUD\n"
            f"─────────────────────────────\n"
            f"Mission Status: {feasibility_status}\n"
            f"DP Optimal Energy: {dp_result.total_energy_wh:.1f} Wh ({dp_result.total_energy_joules / 1000.0:.1f} kJ)\n"
            f"Greedy Energy: {greedy_result.total_energy_wh:.1f} Wh" if greedy_result else ""
        ) + (
            f"{savings_str}\n"
            f"Flight Duration: {dp_result.total_flight_time_s / 60.0:.1f} min\n"
            f"Remaining SoC: {dp_result.remaining_soc_percent:.1f}%\n"
            f"Bitmask DP Time: {dp_result.execution_time_s * 1000.0:.2f} ms"
        )

        ax.text(
            0.02,
            0.98,
            hud_text,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily="monospace",
            color=self.text_color,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=self.card_color,
                edgecolor=feas_color,
                linewidth=1.5,
                alpha=0.92,
            ),
            zorder=11,
        )

        # Styling adjustments
        ax.set_title(title, fontsize=14, fontweight="bold", color=self.text_color, pad=16)
        ax.set_xlabel("Local Tangent Plane East Displacement (meters)", fontsize=10.5, color=self.text_color)
        ax.set_ylabel("Local Tangent Plane North Displacement (meters)", fontsize=10.5, color=self.text_color)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.grid(True, linestyle=":", color=self.grid_color, alpha=0.7)
        ax.tick_params(colors=self.text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(self.grid_color)

        legend = ax.legend(
            loc="lower right",
            facecolor=self.card_color,
            edgecolor=self.grid_color,
            fontsize=9,
            labelcolor=self.text_color,
            framealpha=0.9,
        )

        fig.tight_layout()

        if save_path is not None:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

        return fig
