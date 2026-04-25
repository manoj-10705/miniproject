# Intelligent Warehouse Distribution Network Optimiser

A full-stack application for optimizing distribution networks using machine learning and operations research.

## Features
- **Demand Forecasting**: Random Forest model to predict future demand.
- **Allocation Optimization**: Linear programming for warehouse-to-store allocation.
- **Vehicle Routing**: Optimal route generation using Google OR-Tools.
- **Disruption Simulation**: Tools to test network resilience against demand surges and capacity drops.

## Tech Stack
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts.
- **Backend**: FastAPI (Python), scipy, scikit-learn, Google OR-Tools.

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.10+)

### Installation
1. Install frontend dependencies:
   ```bash
   npm install
   ```
2. Set up Python environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate # or venv\Scripts\activate on Windows
   pip install -r ../requirements.txt
   ```

### Running the App
```bash
npm run dev
```

## Usage
1. Upload the required Excel datasets via the dashboard.
2. View the generated forecasts, allocations, and routes.
3. Use the Scenario Panel to simulate network disruptions.

## License
MIT
