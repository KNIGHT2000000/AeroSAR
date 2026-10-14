# AeroSAR-Optimizer
### Energy-Aware Autonomous UAV Search-and-Rescue Route Optimization via Asymmetric Bitmask Dynamic Programming & Kinematic Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-10%20passed%20(100%25)-brightgreen.svg)](tests/)
[![MAVLink: QGC WPL 110](https://img.shields.io/badge/MAVLink-QGC%20WPL%20110-orange.svg)](output/mission_plan.waypoints)
[![Interactive GCS](https://img.shields.io/badge/UI-Streamlit%20GCS-red.svg)](visualization/dashboard.py)

---

## 1. Executive Summary

**AeroSAR-Optimizer** bridges mathematical algorithm design (Asymmetric Traveling Salesperson Problem solved via Held-Karp Dynamic Programming with Bitmasking) and real-world cyber-physical robotics (aerodynamic wind drift, non-linear quadcopter electrical power draw, battery State-of-Charge safety margins, and MAVLink mission export).

In high-altitude alpine or maritime Search-and-Rescue (SAR) sorties, steady atmospheric wind fields break metric symmetry: flying against a headwind requires significantly more time and electrical energy than flying with a tailwind. Conventional Euclidean TSP heuristics produce suboptimal routes that risk in-flight battery starvation. AeroSAR-Optimizer models directed aerodynamic kinematics, constructs a strictly asymmetric energy matrix $C[i][j]$, computes the provably optimal closed flight tour in $O(n^2 2^n)$ time, evaluates battery feasibility, activates automated priority triage when energy is constrained, and exports production **QGroundControl / MAVLink (.waypoints)** missions ready for autopilot deployment.

---

## 2. Mathematical Foundations & Kinematics

### 2.1 Aerodynamic Drone Power Model
A multirotor UAV experiences non-linear electrical power consumption dominated by hover thrust requirements and parasite aerodynamic drag:

$$P_{\text{flight}} = P_{\text{hover}} + c_{\text{drag}} \cdot \Vert\vec{V}_{\text{air}}\Vert^3$$

- $P_{\text{hover}}$: Baseline mechanical and electrical hover power (Watts).
- $c_{\text{drag}}$: Lumped parasitic drag coefficient ($\text{W} / (\text{m/s})^3$).
- $\vec{V}_{\text{air}}$: Commanded true airspeed vector ($\text{m/s}$).

### 2.2 Kinematic Wind Triangle Drift Compensation
Let the desired ground flight path between Waypoint $i$ ($\vec{X}_i$) and Waypoint $j$ ($\vec{X}_j$) have displacement vector $\vec{D}_{ij} = \vec{X}_j - \vec{X}_i$, with unit ground heading $\hat{u} = \vec{D}_{ij} / \Vert\vec{D}_{ij}\Vert$.

Under horizontal atmospheric wind field $\vec{V}_{\text{wind}} = (V_{w_x}, V_{w_y})$, the drone ground velocity vector is:

$$\vec{V}_{\text{ground}} = \vec{V}_{\text{air}} + \vec{V}_{\text{wind}} = v_g \hat{u}$$

To maintain course along $\hat{u}$, the autopilot commands an airspeed vector $\vec{V}_{\text{air}} = v_g \hat{u} - \vec{V}_{\text{wind}}$ with magnitude equal to nominal airspeed $V_{\text{air}}$:

$$\Vert v_g \hat{u} - \vec{V}_{\text{wind}} \Vert^2 = V_{\text{air}}^2$$

Expanding yields the quadratic equation in ground speed $v_g$:

$$v_g^2 - 2 (\hat{u} \cdot \vec{V}_{\text{wind}}) v_g + (\Vert\vec{V}_{\text{wind}}\Vert^2 - V_{\text{air}}^2) = 0$$

Let $w_\parallel = \hat{u} \cdot \vec{V}_{\text{wind}}$ (parallel component) and $w_\perp = \sqrt{\Vert\vec{V}_{\text{wind}}\Vert^2 - w_\parallel^2}$ (crosswind component):

$$v_g = w_\parallel + \sqrt{V_{\text{air}}^2 - w_\perp^2}$$

- **Crosswind Barrier**: If $w_\perp > V_{\text{air}}$, the drone cannot hold track; the transition is marked infeasible ($C[i][j] = \infty$).
- **Headwind Drift**: If $v_g \le 0$, headwind exceeds airspeed capability ($C[i][j] = \infty$).

### 2.3 Directed Asymmetric Cost Matrix
Transit duration along directed leg $i \to j$ is:

$$t_{\text{transit}}(i, j) = \frac{\Vert\vec{X}_j - \vec{X}_i\Vert}{v_g(i, j)}$$

Total leg energy expenditure combining flight transit and survivor sensor loiter is:

$$C[i][j] = P_{\text{flight}} \cdot t_{\text{transit}}(i, j) + P_{\text{hover}} \cdot t_{\text{loiter}}(j)$$

Because $v_g(i, j) \ne v_g(j, i)$ whenever $\vec{V}_{\text{wind}} \ne \vec{0}$, the matrix is strictly directed and asymmetric:

$$C[i][j] \ne C[j][i]$$

The degree of metric skew is quantified by the normalized **Asymmetry Index $\alpha$**:

$$\alpha = \frac{\sum_{i < j} |C[i][j] - C[j][i]|}{\sum_{i < j} (C[i][j] + C[j][i])}$$

---

## 3. Core Algorithmic Architecture

```
                  ┌────────────────────────────────────────┐
                  │       SAR Mission Configuration        │
                  │   (Datum, Survivors, Wind, Battery)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Kinematic Physics Engine          │
                  │   (Wind Triangle & Cubic Drag Model)   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Directed Asymmetric Cost Matrix     │
                  │       C[i][j] in Joules & Watts        │
                  └─────────┬───────────────────┬──────────┘
                            │                   │
               ┌────────────┴─────┐       ┌─────┴──────────────┐
               ▼                  ▼       ▼                    ▼
     ┌───────────────────┐ ┌─────────────┐ ┌────────────────────────┐
     │ Held-Karp Bitmask │ │ Greedy (NN) │ │ Brute Force Baseline   │
     │ DP Solver         │ │ Heuristic   │ │ O(n!) Permutations     │
     │ O(n² 2ⁿ) Exact    │ │ O(n²) Fast  │ │ (Validation for n<=8)  │
     └─────────┬─────────┘ └─────────────┘ └────────────────────────┘
               │
               ▼
     ┌────────────────────────────────────────────────────────┐
     │ Battery Feasibility Check: E_tour <= E_usable          │
     │   [If Infeasible] -> Automated Priority Triage Engine  │
     └─────────┬──────────────────────────────────────────────┘
               │
               ├──────────────────────────┬──────────────────────────┐
               ▼                          ▼                          ▼
     ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
     │ MAVLink Telemetry │      │ Tactical Quiver   │      │ Streamlit GCS     │
     │ QGC WPL 110 Export│      │ Trajectory Plot   │      │ Real-Time HUD     │
     └───────────────────┘      └───────────────────┘      └───────────────────┘
```

### 3.1 Held-Karp Bitmask DP Formulation
- **State Definition**: $DP[\text{mask}][u]$ is the minimum energy required to start from Base Depot $0$, visit all nodes indicated by bitmask `mask`, and currently end at node $u$.
- **Base Case**: $DP[1][0] = 0.0$ (only node 0 visited; cost 0).
- **Recurrence**:
  $$DP[\text{mask} \mid (1 \ll v)][v] = \min_{u \in \text{mask}} \Big( DP[\text{mask}][u] + C[u][v] \Big)$$
- **Topological Guarantee**: By iterating odd masks `for mask in range(1, 1 << N, 2)`, bit 0 is permanently anchored, and `mask | (1 << v) > mask` always, guaranteeing strictly DAG topological dependency order.
- **Tour Closure & Backtracking**:
  $$\text{Cost}^* = \min_{u \in [1, N-1]} \Big( DP[(1 \ll N) - 1][u] + C[u][0] \Big)$$
  Parent backtracking reconstructed using bitwise operations: `mask = mask ^ (1 << curr)`.
- **Complexity**: $O(n^2 2^n)$ time, $O(n 2^n)$ space.

### 3.2 Battery Envelope & Automated Triage
The usable battery envelope is defined as:

$$E_{\text{usable}} = E_{\text{capacity}} \times (1 - \text{reserve\_margin})$$

If the minimum-energy tour requires $E_{\text{tour}} > E_{\text{usable}}$, the mission is flagged **INFEASIBLE**. The system initiates the automated **Triage Contingency Engine**:
1. Beacons are indexed by triage urgency priority score $w_i \in [1.0, 10.0]$ (e.g., hypothermia risk, avalanche transceiver strength).
2. The solver prunes lowest priority targets iteratively and re-solves DP until a valid closed tour fits within the usable battery envelope.
3. Both nominal requirements, deficit metrics, and recommended triage sorties are logged for the operator.

### 3.3 Theoretical Complexity & Problem Classification

To fulfill rigorous Design and Analysis of Algorithms (DAA) criteria, the computational complexity is formally classified:

#### A. Decision Problem Formulation ($\text{SAR-DECISION}$)
> **Instance**: A set of $N$ waypoints $V = \{0, 1, \dots, N-1\}$, a directed asymmetric cost matrix $C[u][v] \in \mathbb{R}^+$, and an energy budget threshold $B = E_{\text{usable}}$.
> **Question**: Does there exist a cyclic permutation (closed tour) $\pi$ of $V$ starting and ending at Base Depot 0 such that:
> $$\sum_{i=0}^{N-1} C[\pi(i)][\pi(i+1)] \le B \quad \text{where } \pi(N) = \pi(0) = 0?$$

#### B. Membership in NP
Given a proposed flight certificate (a permutation sequence $\pi$), a deterministic verifier:
1. Checks that $\pi$ contains every waypoint in $V$ exactly once in $O(N)$ time.
2. Sums the directed leg energy costs along $\pi$ and checks if $\sum C \le B$ in $O(N)$ time.
Since verification completes in deterministic polynomial time ($O(N)$), $\text{SAR-DECISION} \in \mathbf{NP}$.

#### C. NP-Completeness Proof (Reduction from Directed Hamiltonian Cycle)
We establish a polynomial-time reduction from the known NP-Complete **Directed Hamiltonian Cycle (DHC)** problem ($\text{DHC} \le_p \text{SAR-DECISION}$):
1. Given a directed graph $G = (V_G, E_G)$ with $|V_G| = N$, construct an asymmetric cost matrix $C$:
   $$C[u][v] = \begin{cases} 1 & \text{if } (u, v) \in E_G \\ \infty & \text{if } (u, v) \notin E_G \end{cases}$$
2. Set budget $B = N$.
3. Clearly, $G$ contains a Directed Hamiltonian Cycle if and only if $\text{SAR-DECISION}$ returns **YES** with total cost $\le N$.
4. The reduction runs in $O(N^2)$ polynomial time.
Therefore, the decision version of the energy-aware routing problem is **NP-Complete**.

#### D. NP-Hardness of the Optimization Problem ($\text{SAR-OPTIMIZE}$)
Because any polynomial-time algorithm that solves the optimization version ($\min E_{\text{tour}}$) could solve $\text{SAR-DECISION}$ by simply checking if $\min E_{\text{tour}} \le B$, the search-and-rescue tour optimization problem is strictly **NP-Hard**.

#### E. Reduction Chain from Boolean Satisfiability (SAT)
$$\text{3-SAT} \;\le_p\; \text{Directed Hamiltonian Cycle} \;\le_p\; \text{Directed ATSP Decision} \;\le_p\; \text{AeroSAR-Optimizer}$$

---

## 4. Project Directory Tree

```
aerosar_optimizer/
├── README.md                 # Complete documentation and mathematical proofs
├── requirements.txt          # Python dependencies (numpy, scipy, matplotlib, streamlit, ...)
├── config.yaml               # Mission, drone physics, wind, and battery parameters
├── main.py                   # Executable end-to-end pipeline (< 5 seconds execution)
├── core/
│   ├── __init__.py           # Exported package symbols
│   ├── physics.py            # Waypoint, WindField, kinematic drift & battery model
│   ├── cost_matrix.py        # Directed asymmetric energy matrix C[i][j] calculator
│   ├── brute_force.py        # O(n!) permutation baseline & Greedy nearest-neighbor
│   ├── bitmask_dp.py         # Held-Karp O(n² 2ⁿ) bitmask DP & automated triage engine
│   └── telemetry.py          # QGroundControl WPL 110 MAVLink mission waypoint generator
├── simulation/
│   ├── __init__.py           # Simulation exports
│   ├── synthetic_data.py     # Clustered avalanche/maritime survivor beacon generator
│   └── runner.py             # Automated comparative benchmarking suite
├── visualization/
│   ├── __init__.py           # Visualization exports
│   ├── plotter.py            # Tactical trajectory visualizer with aerodynamic quivers
│   └── dashboard.py          # Interactive Streamlit Ground Control Station HUD
├── tests/
│   └── test_aerosar.py       # Comprehensive pytest suite (10/10 passing)
└── output/                   # Auto-generated runtime artifacts
    ├── mission_trajectory.png# High-res trajectory map with wind vectors
    ├── mission_plan.waypoints# Production QGroundControl mission file
    └── benchmark_results.csv # Scalability performance data
```

---

## 5. Quickstart & Usage

### 5.1 Environment Setup
```bash
# Clone or navigate to repository root
cd d:/DAA_Project

# Install minimal dependencies
pip install -r aerosar_optimizer/requirements.txt
```

### 5.2 Execute End-to-End Pipeline (< 5 Seconds)
Run the complete production pipeline from the repository root:
```bash
python main.py
```
Or from inside `aerosar_optimizer/`:
```bash
cd aerosar_optimizer
python main.py
```

**Console Output:**
```text
========================================================================================
     ___                    ____    _    ____          ___        _   _           _             
    / _ \  ___ _ __ ___    / ___|  / \  |  _ \        / _ \ _ __ | |_(_)_ __ ___ (_)_______ _ __ 
   / /_\ \/ _ \ '__/ _ \   \___ \ / _ \ | |_) |_____ | | | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
  / /_\\ \  __/ | | (_) |   ___) / ___ \|  _ <|_____|| |_| | |_) | |_| | | | | | | |/ /  __/ |   
 /_/   \_\\___|_|  \___/   |____/_/   \_\_| \_\       \___/| .__/ \__|_|_| |_| |_|_/___\___|_|   
                                                           |_|                                    
 Energy-Aware Autonomous UAV SAR Route Optimization via Asymmetric Bitmask Dynamic Programming
========================================================================================

[*] Initializing Operational Parameters:
    - Drone Model:            HexaRotor-SAR-Heavy
    - Nominal Airspeed:       15.0 m/s (54.0 km/h)
    - Atmospheric Wind:       6.8 m/s @ 55.0° (Vx=+3.90, Vy=+5.57 m/s)
    - Battery Gross Capacity: 350.0 Wh (1260.0 kJ)
    - Usable Mission Buffer:  280.0 Wh (Reserve Margin: 20%)
    - Search Theater Radius:  2500 m around Base Camp (46.5580°N, 8.5600°E)

[+] Generated 10 Survivor Beacons in Clustered Avalanche Zone.
[+] Computed 11x11 Directed Asymmetric Cost Matrix in 3.10 ms
    - Asymmetry Metric (alpha): 0.2627 (Strictly Asymmetric: True)
    - Peak Pairwise Leg Skew:   162.6%

========================================================================================
                      MISSION OPTIMIZATION TELEMETRY SUMMARY
========================================================================================
Metric                           | Bitmask Held-Karp DP     | Greedy Nearest-Neighbor 
----------------------------------------------------------------------------------------
Optimal Closed Tour              | [0, 9, 3, 6, 2, 5, 8, 1, 10, 4, 7, 0] | [0, 6, 3, 9, 7, 1, 10, 4, 2, 5, 8, 0]
Total Energy (Watt-hours)        | 85.91                    | 93.36                   
Total Energy (Kilojoules)        | 309.29                   | 336.08                  
Sortie Duration (minutes)        | 11.27                    | 12.22                   
Remaining Battery SoC (%)        | 75.5                     | 73.3                    
Feasibility Status               | FEASIBLE                 | FEASIBLE                
Algorithm Wall Runtime           | 74.38 ms                 | 0.05 ms                 
State Space Evaluations          | 23,050                   | 100                     
----------------------------------------------------------------------------------------
[*] DP Aerodynamic Energy Savings: 7.44 Wh (8.0% reduction over Greedy)

[+] Exported MAVLink Mission File: output/mission_plan.waypoints
    - Compatible with: QGroundControl, ArduPilot, PX4 Autopilot, Mission Planner
[+] Saved High-Resolution Trajectory Map: output/mission_trajectory.png
[✓] Complete AeroSAR-Optimizer Pipeline Executed in 3.51 seconds (< 5.00s limit).
```

### 5.3 Launch Streamlit Ground Control Station Dashboard
Launch the interactive web UI:
```bash
streamlit run aerosar_optimizer/visualization/dashboard.py
```
Features available in the dashboard:
- Interactive sliders for **Survivor Count ($n \in [4, 15]$)**, **Wind Speed & Direction**, **Nominal Airspeed**, and **Battery Capacity**.
- Live tactical 2D map with wind quivers contrasting DP optimal vs Greedy.
- Real-time battery State-of-Charge depletion gauge and reserve margin buffer.
- Automated triage alert panel with pruned contingency routes.
- One-click **Download button** for the QGroundControl `.waypoints` mission file.

### 5.4 Run Algorithmic Benchmark Suite
To execute the scalability benchmark suite up to $n=14$:
```bash
python aerosar_optimizer/main.py --full-benchmark
# or
python aerosar_optimizer/simulation/runner.py
```

### 5.5 Run Test Suite
```bash
pytest aerosar_optimizer/tests/test_aerosar.py -v
```

---

## 6. Empirical Benchmark Results

Measured on AMD Ryzen / Intel Core CPU under 6.8 m/s wind field:

| Targets ($n$) | States / Permutations | Brute Force $O(n!)$ | Bitmask DP $O(n^2 2^n)$ | Speedup Factor | DP Energy (Wh) | Greedy Gap (%) | DP Memory (KB) | Mathematical Verification |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** | 24 | 0.00026 s | 0.00028 s | 0.9x | 38.6 | +0.0% | 1.6 KB | **MATCH (Exact)** |
| **6** | 720 | 0.00959 s | 0.00222 s | 4.3x | 68.9 | +4.7% | 8.8 KB | **MATCH (Exact)** |
| **8** | 40,320 | 0.81711 s | 0.01238 s | 66.0x | 71.6 | +3.8% | 45.0 KB | **MATCH (Exact)** |
| **10** | 3,628,800 | 54.01 s * | 0.07455 s | **724.5x** | 85.9 | +8.7% | 220.0 KB | Verified against DP |
| **12** | 479,001,600 | 6,810.1 s * | 0.42706 s | **15,946x** | 87.8 | +9.5% | 1,040 KB | Verified against DP |
| **14** | 87,178,291,200 | 1,239,437 s * (~14 days) | 2.48813 s | **498,140x** | 90.3 | +10.7% | 4,800 KB | Verified against DP |

*\* Extrapolated $O(n!)$ runtime based on empirical single-evaluation CPU baseline.*

---

## 7. MAVLink / QGroundControl Mission File Format

Exported mission files comply with **QGC WPL 110**:
```text
QGC WPL 110
0	1	0	16	0.000000	0.000000	0.000000	0.000000	46.5580000	8.5600000	0.0	1
1	0	3	22	0.000000	0.000000	0.000000	0.000000	46.5580000	8.5600000	50.0	1
2	0	3	16	5.000000	3.000000	0.000000	0.000000	46.5612140	8.5721900	50.0	1
...
12	0	3	20	0.000000	0.000000	0.000000	0.000000	0.0000000	0.0000000	0.0	1
```
- **Line 0**: Base Home reference (Command 16).
- **Line 1**: Autonomous climb to survey altitude (50 m AGL, Command 22 `NAV_TAKEOFF`).
- **Lines 2..K**: Optimized sequence of survivor waypoints with 5.0 s thermal camera loiter and 3.0 m acceptance radius (Command 16 `NAV_WAYPOINT`).
- **Line K+1**: Automatic Return to Launch (Command 20 `NAV_RETURN_TO_LAUNCH`).

---

## 8. License

MIT License. Designed and engineered for production autonomous aerospace and robotics systems.
