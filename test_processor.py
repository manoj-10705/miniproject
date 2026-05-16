from backend.utils.data_processor import DataProcessor
import json

dp = DataProcessor()
data = dp.load_local_data("data")
print(f"Nodes extracted: {len(data['demand_nodes'])}")
print(f"Regions with distances: {list(data['distance_matrices'].keys())}")
print(f"Locations extracted: {len(data['locations'])}")
