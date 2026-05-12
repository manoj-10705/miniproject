from utils.data_processor import DataProcessor
from models.demand_forecasting import DemandForecaster
from models.allocation_optimizer import AllocationOptimizer
from models.vehicle_routing import VehicleRouter

def test_pipeline():
    print("Loading data...")
    dp = DataProcessor()
    processed_data = dp.load_local_data("../data")
    print(f"Data loaded: {len(processed_data['locations'])} locations")

    print("Training Demand Forecaster...")
    df = DemandForecaster()
    df.train(processed_data)
    forecast = df.forecast(processed_data)
    print("Forecast generated.")

    print("Running Allocation Optimizer...")
    warehouses = [loc for loc in processed_data.get("locations", []) if loc["type"] == "warehouse"]
    stores = [loc for loc in processed_data.get("locations", []) if loc["type"] == "store"]

    ao = AllocationOptimizer()
    allocation_result = ao.optimize(
        warehouses=warehouses,
        stores=stores,
        demand_forecast=forecast,
        capacity_data=processed_data.get("capacity", {}),
        cost_data=processed_data.get("costs", {})
    )
    print(f"Allocations generated: {len(allocation_result.get('allocations', {}))} warehouses used.")

    print("Running Vehicle Routing...")
    vr = VehicleRouter()
    routing_result = vr.optimize_routes(
        allocation_result=allocation_result,
        locations=processed_data.get("locations", []),
        distance_matrices=processed_data.get("distance_matrices", {}),
        time_matrices=processed_data.get("time_matrices", {})
    )
    print(f"Routing generated: {routing_result['n_vehicles_used']} vehicles used.")
    print("Pipeline Test SUCCESS")

if __name__ == "__main__":
    test_pipeline()
