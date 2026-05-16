import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class DataProcessor:
    """Enhanced Data Processor for handling multi-sheet regional data"""

    def __init__(self):
        self.regions = ["Trabzon", "Rize", "Giresun", "Ordu"]

    def load_local_data(self, data_dir: str = "data") -> Dict[str, Any]:
        """Loads data directly from the local filesystem (v3 structure)."""
        import os
        raw_data = {}
        for file in os.listdir(data_dir):
            if file.endswith(".xlsx"):
                try:
                    df_dict = pd.read_excel(os.path.join(data_dir, file), sheet_name=None)
                    raw_data[file] = df_dict
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        return self.process_all_data(raw_data)

    def process_all_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Process all raw excel data dictionaries"""
        logger.info("Starting data processing")
        processed = {}

        for filename, data in raw_data.items():
            fname = filename.lower().replace("_", "")
            if "geolocations" in fname:
                processed["locations"] = self._process_geolocations(data)
            elif "demand" in fname:
                processed["demand"] = self._process_demand(data)
            elif "distance" in fname:
                processed["distance"] = self._process_matrices(data, "Distance")
            elif "time" in fname:
                processed["time"] = self._process_matrices(data, "Time")
            elif "capacity" in fname:
                processed["capacity"] = self._process_capacity(data)
            elif "costcluster" in fname or "cost" in fname:
                processed["costs"] = self._process_costs(data)

        # Merge and format for ML models
        formatted_data = self._format_for_ml(processed)
        return formatted_data

    def _process_geolocations(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        locations = []
        for sheet_name, df in data.items():
            for i, row in df.iterrows():
                loc_type = "warehouse" if "WH" in sheet_name else "store"
                try:
                    lat, lon = float(row["lat"]), float(row["lon"])
                    locations.append({
                        "id": f"{sheet_name}_{i}",
                        "region": sheet_name.replace("S-WH ", "").replace("P-WH", "Central"),
                        "type": loc_type,
                        "lat": lat,
                        "lon": lon
                    })
                except (ValueError, KeyError):
                    continue
        return locations

    def _process_demand(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        records = []
        for sheet_name, df in data.items():
            region = sheet_name.replace("Demand ", "")
            df = df.dropna(how="all").reset_index(drop=True)
            if df.empty: continue

            # First column is usually index/node
            id_col = df.columns[0]
            for i, row in df.iterrows():
                node_id = str(row[id_col])
                for col in df.columns[1:]:
                    try:
                        val = float(row[col])
                        records.append({
                            "node_id": f"{region}_{node_id}",
                            "region": region,
                            "time_step": str(col),
                            "demand": val
                        })
                    except (ValueError, TypeError):
                        pass
        return records

    def _process_matrices(self, data: Dict[str, pd.DataFrame], matrix_type: str) -> Dict[str, Any]:
        matrices = {}
        for sheet_name, df in data.items():
            region = sheet_name.replace(f"{matrix_type} ", "")
            df_clean = df.loc[:, ~df.columns.str.lower().str.contains("unnamed|nan")]
            numeric_df = df_clean.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                matrices[region] = numeric_df.values.tolist()
        return matrices

    def _process_capacity(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        capacities = {}
        for sheet_name, df in data.items():
            region = sheet_name.replace("Capacity ", "")
            if df.empty: continue

            # Sum up total capacity across all warehouses for the region as a simple baseline
            try:
                # Row 0 is usually total 'capacity'
                cap_row = df.iloc[0, 1:]
                total_cap = pd.to_numeric(cap_row, errors='coerce').sum()
                capacities[region] = float(total_cap)
            except Exception:
                capacities[region] = 10000.0
        return capacities

    def _process_costs(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        costs = {}
        for sheet_name, df in data.items():
            region = sheet_name.replace("Cost ", "")
            df_clean = df.loc[:, ~df.columns.str.lower().str.contains("unnamed|nan")]
            numeric_df = df_clean.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                costs[region] = numeric_df.values.tolist()
        return costs

    def _format_for_ml(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard dictionaries into ML-ready formats (Adjacency matrices, Time Series)"""
        formatted = {"locations": processed.get("locations", [])}
        
        # 1. Structure Demand into Time-Series Matrix
        demand_df = pd.DataFrame(processed.get("demand", []))
        if not demand_df.empty:
            # Pivot to get nodes x time_steps
            pivot_df = demand_df.pivot_table(index="node_id", columns="time_step", values="demand", aggfunc="mean").fillna(0)
            formatted["demand_series"] = pivot_df.values.tolist()
            formatted["demand_nodes"] = pivot_df.index.tolist()
        else:
            formatted["demand_series"] = []
            formatted["demand_nodes"] = []

        # 2. Graph Adjacency
        formatted["distance_matrices"] = processed.get("distance", {})
        formatted["time_matrices"] = processed.get("time", {})
        
        # 3. Optimization constraints
        formatted["capacity"] = processed.get("capacity", {})
        formatted["costs"] = processed.get("costs", {})

        return formatted
