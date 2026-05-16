import torch
from backend.models.demand_forecasting import DemandForecaster

df = DemandForecaster()
dummy_data = {
    "demand_series": [[10, 12, 14, 13, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27, 28, 30] * 2],
    "demand_nodes": ["node1", "node2"],
    "distance_matrices": {}
}

# The array has shape (1, 32). We'll make it (2, 16) to match the two nodes.
dummy_data["demand_series"] = [
    [10, 12, 14, 13, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27, 28, 30],
    [11, 13, 15, 14, 16, 17, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31]
]

res = df.train(dummy_data)
print("Training result:", res)
f_res = df.forecast(dummy_data)
print("Forecast result:", f_res)
