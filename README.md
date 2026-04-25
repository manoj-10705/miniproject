# Intelligent Warehouse-to-Store Distribution Network Optimiser

A full-stack research platform for pharmaceutical supply chain logistics, combining Random Forest demand forecasting, Linear Programming allocation, and Capacitated Vehicle Routing.

## 🎯 Project Overview
This system implements a three-stage intelligence pipeline to optimize distribution across regional clusters:
1. **Demand Forecasting**: Predicts future demand using Random Forest regression.
2. **Allocation Optimization**: Minimizes distribution costs subject to warehouse capacity using Linear Programming.
3. **Vehicle Routing**: Generates optimal delivery routes using Google OR-Tools with Guided Local Search.
4. **Resilience Testing**: Simulates network disruptions (demand surges, capacity drops) to evaluate system stability.

## 🛠 Technical Stack
- **Frontend**: React 19, TypeScript, Vite, Recharts, TailwindCSS.
- **Backend**: FastAPI, Uvicorn, Python 3.10+.
- **Data Science**: Scikit-learn, SciPy, Pandas, NumPy.
- **Optimization**: Google OR-Tools (CVRP).

## 📊 Dataset & Metrics
The project utilizes real pharmaceutical distribution data from 4 regional clusters (Trabzon, Rize, Ordu, Giresun). 
- **Feasibility Boundary**: Empirically derived at `α/γ ≤ 2.31`.
- **Scale**: 1,379 customer nodes across 12 temporal periods.

## ⚙️ Setup & Installation

### 1. Frontend
```bash
npm install
npm run dev
```

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate # Windows
pip install -r ../requirements.txt
python main.py
```

## 📁 Project Structure
- `src/`: React frontend components and logic.
- `backend/`: FastAPI server and optimization models.
- `backend/models/`: Demand forecasting, allocation, and routing logic.
- `mnpaper.pdf`: Supporting research paper.

---
*Author: Manoj*  
*License: MIT*
