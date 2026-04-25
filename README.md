<div align="center">

# 🏭 Intelligent Warehouse-to-Store Distribution Network Optimiser with Disruption Rerouting

### A Hybrid ML/OR Supply Chain Intelligence Platform

[![Author](https://img.shields.io/badge/Author-Manoj%20(manoj--10705)-blue?style=for-the-badge&logo=github)](https://github.com/manoj10705)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

<br/>

**Designed, Architected, and Developed by [Manoj](https://github.com/manoj10705)**

*Full-stack research platform combining Random Forest demand forecasting, Linear Programming allocation, and Capacitated Vehicle Routing with adversarial disruption simulation — built on real pharmaceutical distribution data from the Eastern Black Sea region of Turkey.*

---

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Technical Stack](#-technical-stack)
- [Core Intelligence Pipeline](#-core-intelligence-pipeline)
- [Dataset Specifications](#-dataset-specifications)
- [Key Results & Metrics](#-key-results--metrics)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Research Paper](#-research-paper)
- [Author & Contact](#-author--contact)
- [License](#-license)

---

## 🎯 Project Overview

This platform addresses a multi-objective optimization problem in pharmaceutical supply chain logistics. Unlike traditional single-stage optimizers, this system implements a **three-stage sequential intelligence pipeline**:

1. **Stage 1 — Demand Forecasting**: Random Forest regression with engineered temporal features (lag, rolling mean, exponential smoothing) to predict future demand across 1,379 customer nodes distributed across 4 regional clusters (Trabzon, Rize, Ordu, Giresun).

2. **Stage 2 — Allocation Optimization**: Linear Programming (LP) formulation minimizing total distribution cost subject to warehouse capacity constraints and store demand satisfaction constraints, using `scipy.optimize.linprog`.

3. **Stage 3 — Vehicle Routing**: Capacitated Vehicle Routing Problem (CVRP) solved via Google OR-Tools with Guided Local Search metaheuristic, generating optimal delivery routes per warehouse cluster.

**The Disruption Layer**: The system injects adversarial perturbations — demand surges (α), capacity reductions (γ), and route blockages (β) — to stress-test network resilience. A formal **feasibility boundary** at `α/γ ≤ 2.31` was empirically derived, beyond which no feasible allocation exists.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA CONTEXT LAYER                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ GeoLocations │  │ DemandCluster    │  │ CapacityClustered        │  │
│  │ .xlsx        │  │ .xlsx            │  │ .xlsx                    │  │
│  │ (4 regions,  │  │ (1,379 customers │  │ (Warehouse caps per      │  │
│  │  273 nodes)  │  │  x 12 periods)   │  │  cluster & period)       │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
│         │                   │                         │                │
│  ┌──────┴───────┐  ┌───────┴──────────┐  ┌───────────┴─────────────┐  │
│  │ Distance     │  │ Time.xlsx        │  │ CostCluster.xlsx        │  │
│  │ Cluster.xlsx │  │ (P2P transit     │  │ CostsMWC-Clustered.xlsx │  │
│  │ (OD matrix)  │  │  times, 101 pts) │  │ (per-unit costs)        │  │
│  └──────────────┘  └──────────────────┘  └─────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CORE INTELLIGENCE ENGINE                             │
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │  Stage 1: ML        │    │  Stage 2: LP        │                     │
│  │  ─────────────────  │    │  ─────────────────  │                     │
│  │  Random Forest      │──▶│  scipy.linprog      │                     │
│  │  Demand Forecaster  │ d̂ │  Allocation         │                     │
│  │  (demand_forecas-   │    │  (allocation_opti-  │                     │
│  │   ting.py, 10KB)    │    │   mizer.py, 18KB)   │                     │
│  └─────────────────────┘    └──────────┬──────────┘                     │
│                                        │ x_ws                           │
│                                        ▼                                │
│                             ┌─────────────────────┐                     │
│                             │  Stage 3: CVRP      │                     │
│                             │  ─────────────────  │                     │
│                             │  OR-Tools GLS       │                     │
│                             │  Vehicle Routing    │                     │
│                             │  (vehicle_routing   │                     │
│                             │   .py, 18KB)        │                     │
│                             └──────────┬──────────┘                     │
│                                        │                                │
└────────────────────────────────────────┼────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     RESILIENCE TESTING LAYER                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Disruption Injector (scenario_simulator.py)                     │   │
│  │  ──────────────────────────────────────────                      │   │
│  │  α: Demand Surge Multiplier   (1.0x → 3.0x)                     │   │
│  │  γ: Capacity Reduction Factor (0% → 80%)                        │   │
│  │  β: Route Blockage Injection  (Random edge removal)              │   │
│  │                                                                  │   │
│  │  Feasibility Boundary: α/γ ≤ 2.31                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Baseline Metrics ◄──── Compare ────► Disrupted Metrics                │
└─────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FRONTEND DASHBOARD (React 19 + Recharts)          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  FileUpload   │  │  Dashboard   │  │  ScenarioPanel              │  │
│  │  (.tsx, 7KB)  │  │  (.tsx, 18KB)│  │  (.tsx, 14KB)               │  │
│  │  Drag & Drop  │  │  KPI Cards   │  │  What-If Simulator          │  │
│  │  7-file ingest│  │  Cost Charts │  │  α/γ/β Sliders              │  │
│  │              │  │  Route Viz   │  │  Side-by-side comparison     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19, TypeScript, Vite 6 | Interactive SPA dashboard |
| **Charting** | Recharts 3.5 | KPI cards, cost breakdowns, route visualization |
| **Styling** | TailwindCSS 3 | Responsive layout, dark-mode capable |
| **Backend** | FastAPI, Uvicorn | REST API for optimization pipeline |
| **ML** | scikit-learn (RandomForestRegressor) | Demand time-series forecasting |
| **Optimization** | scipy.optimize.linprog | LP-based allocation under constraints |
| **Routing** | Google OR-Tools (pywrapcp) | CVRP with Guided Local Search |
| **Data** | pandas, numpy | Multi-sheet Excel ingestion & transformation |
| **File I/O** | react-dropzone, PapaParse | Client-side drag-and-drop upload |

---

## 🧠 Core Intelligence Pipeline

### Stage 1: Demand Forecasting (`demand_forecasting.py` — 10KB)

The forecasting module engineers temporal features from raw demand time-series:

- **Lag Features**: `demand(t-1)`, `demand(t-2)`, `demand(t-3)` for each store
- **Rolling Statistics**: 3-period rolling mean and standard deviation
- **Exponential Smoothing**: EWM with span=3 for trend capture
- **Model**: `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Train/Test**: 80/20 temporal split preserving time ordering
- **Output**: Per-store demand estimates `d̂_s` fed into Stage 2

### Stage 2: Allocation Optimization (`allocation_optimizer.py` — 18KB)

A linear program minimizing total distribution cost:

```
minimize    Σ_w Σ_s  c_ws · x_ws
subject to  Σ_s x_ws  ≤  Cap_w      ∀ warehouse w
            Σ_w x_ws  ≥  d̂_s        ∀ store s
            x_ws ≥ 0
```

Where `c_ws` is the per-unit cost from warehouse `w` to store `s`, derived from the Cost and CostsMWC Excel sheets. Capacity constraints are loaded from `CapacityClustered.xlsx` per regional cluster (Trabzon, Rize, Ordu, Giresun).

### Stage 3: Vehicle Routing (`vehicle_routing.py` — 18KB)

Capacitated VRP solved using Google OR-Tools:

- **Distance Matrix**: Loaded from `DistanceCluster.xlsx` (origin-destination pairs)
- **Time Matrix**: Point-to-point transit times from `Time.xlsx` (101 pharmacy nodes)
- **Solver**: `routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC` with `LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH`
- **Time Limit**: 30 seconds per solve
- **Output**: Optimal routes per warehouse, total distance/cost

---

## 📊 Dataset Specifications

The project uses **real pharmaceutical distribution data** from the Eastern Black Sea region of Turkey:

| File | Contents | Scale |
|------|----------|-------|
| `GeoLocations.xlsx` | Lat/long coordinates, node types | 4 regions, 273 nodes |
| `DemandCluster.xlsx` | Historical demand per store per period | 1,379 customers × 12 periods |
| `CapacityClustered.xlsx` | Warehouse capacity by cluster | 4 regional warehouses |
| `DistanceCluster.xlsx` | Origin-destination distance matrix | Full pairwise distances |
| `Time.xlsx` | Point-to-point transit times | 101 pharmacy nodes |
| `CostCluster.xlsx` | Per-unit transport costs | Per cluster pair |
| `CostsMWC-Clustered.xlsx` | Multi-warehouse costs | Cross-cluster allocation costs |

### Regional Network Scale (Ground Truth)
| Region | Customers | Total Demand (units) | Active Pharmacies |
|--------|-----------|---------------------|-------------------|
| Trabzon | 363 | 1,206,839 | 101 |
| Rize | 350 | 461,733 | 40 |
| Ordu | 338 | 1,235,007 | 72 |
| Giresun | 328 | 547,579 | 60 |

---

## 📈 Key Results & Metrics

- **Demand Forecast Accuracy**: Random Forest achieved competitive MAPE on temporal holdout
- **Capacity Utilization**: Ranges from 9.84% (Trabzon — over-provisioned) to 51.89% (Ordu — near optimal)
- **Feasibility Boundary**: Empirically derived at `α/γ ≤ 2.31` — beyond this ratio, no feasible allocation exists under simultaneous demand surge and capacity reduction
- **Cost Sensitivity**: 0.0012 cost units/km for representative Rize cluster routes

---

## ⚙️ Installation & Setup

### Prerequisites
- **Node.js** ≥ 18.x
- **Python** ≥ 3.10
- **pip** (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/manoj10705/miniproject.git
cd miniproject
```

### 2. Install Frontend Dependencies
```bash
npm install
```

### 3. Set Up Python Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r ../requirements.txt
cd ..
```

### 4. Run the Application
```bash
npm run dev
```
This starts both the Vite dev server (port 5173) and the FastAPI backend (port 8000) concurrently.

### 5. Upload Data
Open `http://localhost:5173` and drag-and-drop all 7 Excel files into the upload zone. The system will automatically:
1. Parse multi-sheet Excel data
2. Train the Random Forest demand forecaster
3. Solve the LP allocation
4. Generate CVRP routes
5. Display the full dashboard with KPI metrics

---

## 📁 Project Structure

```
miniproject/
├── src/                          # React Frontend
│   ├── App.tsx                   # Main application shell
│   ├── main.tsx                  # React DOM entry point
│   ├── types.ts                  # TypeScript interfaces
│   ├── index.css                 # Global styles (Tailwind)
│   └── components/
│       ├── Dashboard.tsx         # KPI cards, charts, metrics (18KB)
│       ├── FileUpload.tsx        # Drag-and-drop file upload (7KB)
│       └── ScenarioPanel.tsx     # Disruption scenario simulator (14KB)
│
├── backend/                      # FastAPI Backend
│   ├── main.py                   # API routes & orchestration (13KB)
│   ├── models/
│   │   ├── demand_forecasting.py # Random Forest forecaster (10KB)
│   │   ├── allocation_optimizer.py # LP allocation solver (18KB)
│   │   └── vehicle_routing.py    # CVRP solver via OR-Tools (18KB)
│   └── utils/
│       ├── data_processor.py     # Excel multi-sheet parser
│       └── scenario_simulator.py # Disruption injection engine
│
├── data/                         # Processed data artifacts
├── figures/                      # Generated visualizations
├── *.xlsx                        # Raw dataset files (7 files)
├── mnpaper.pdf                   # Research paper
├── package.json                  # Frontend dependencies
├── requirements.txt              # Python dependencies
├── vite.config.ts                # Vite build configuration
└── README.md                     # This file
```

---

## 📄 Research Paper

This project is accompanied by a formal research paper (`mnpaper.pdf`) documenting the methodology, mathematical formulations, experimental setup, and results. The paper includes:

- Formal LP and CVRP mathematical formulations with constraint definitions
- Feature engineering pipeline for temporal demand forecasting
- Adversarial disruption simulation framework
- Empirical derivation of the feasibility boundary theorem
- Comparative analysis of baseline vs. disrupted network performance

---

## 👤 Author & Contact

<table>
<tr>
<td align="center">

**Manoj**

*Sole Developer & Architect*

[![GitHub](https://img.shields.io/badge/GitHub-manoj10705-181717?style=flat-square&logo=github)](https://github.com/manoj10705)
[![Email](https://img.shields.io/badge/Email-ganymede323%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:ganymede323@gmail.com)

</td>
</tr>
</table>

> **Note**: This entire project — architecture design, ML pipeline, optimization solvers, frontend dashboard, data processing, research paper, and all associated code — was conceived, designed, and implemented solely by [Manoj](https://github.com/manoj10705). All commits in the git history are authored by `manoj-10705`. No external contributors.

---

## 📜 License

Copyright © 2025-2026 **Manoj (manoj-10705)**. All rights reserved.

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

*Built with ☕ and determination by [Manoj](https://github.com/manoj10705)*

`commit 9211e4f9 — First commit — Dec 9, 2025 | commit 5a3009ca — models — Jan 19, 2026`

</div>
