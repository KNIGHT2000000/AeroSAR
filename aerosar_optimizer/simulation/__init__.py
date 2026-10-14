"""AeroSAR-Optimizer: Simulation & Benchmarking Package.

Provides synthetic SAR mission generator, clustered survivor beacon positioning,
atmospheric wind field modeling, and high-performance comparative benchmarks.
"""

from .synthetic_data import SyntheticScenarioGenerator, SARScenario
from .runner import BenchmarkRunner, BenchmarkSummary
from .live_simulation import (
    KinematicFlightSimulator,
    DualDroneRaceEngine,
    DroneFlightSnapshot,
    FlightLegTimeline,
)

__all__ = [
    "SyntheticScenarioGenerator",
    "SARScenario",
    "BenchmarkRunner",
    "BenchmarkSummary",
    "KinematicFlightSimulator",
    "DualDroneRaceEngine",
    "DroneFlightSnapshot",
    "FlightLegTimeline",
]
