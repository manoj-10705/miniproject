import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and validate uploaded Excel data"""
    
    def __init__(self):
        self.processed_data = {}
        
    def process_all_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all uploaded data files"""
        try:
            logger.info("Starting data processing")
            
            processed = {}
            
            # Process each data type
            processed['locations'] = self._process_geo_locations(raw_data.get('geo_locations', {}))
            processed['capacity_data'] = self._process_capacity_data(raw_data.get('capacity', {}))
            processed['demand_data'] = self._process_demand_data(raw_data.get('demand', {}))
            processed['distance_matrix'] = self._process_distance_data(raw_data.get('distance', {}))
            processed['time_matrix'] = self._process_time_data(raw_data.get('time', {}))
            processed['cost_data'] = self._process_cost_data(raw_data.get('cost', {}))
            processed['costs_mwc'] = self._process_costs_mwc_data(raw_data.get('costs_mwc', {}))
            
            # Derive warehouses and stores from locations
            warehouses, stores = self._separate_warehouses_stores(processed['locations'])
            processed['warehouses'] = warehouses
            processed['stores'] = stores
            
            # Validate data consistency
            self._validate_data_consistency(processed)
            
            logger.info("Data processing completed successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _process_geo_locations(self, geo_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process geographical locations data"""
        try:
            locations = []

            # Process all sheets in geo locations file
            for sheet_name, df in geo_data.items():
                logger.info(f"Processing geo locations sheet: {sheet_name}")

                # We know from the Excel: columns like "Geo ID", "Latitude", "Longitude"
                cols_lower = {str(c).lower(): c for c in df.columns}

                lat_cols = [cols_lower[k] for k in cols_lower.keys() if k.startswith("latitude") or k == "lat"]
                lon_cols = [cols_lower[k] for k in cols_lower.keys() if k.startswith("longitude") or k in ("lon", "lng")]
                id_cols  = [cols_lower[k] for k in cols_lower.keys() if "geo id" in k or (k.endswith("id") and "grid" not in k)]

                for i, row in df.iterrows():
                    location = {
                        "sheet": sheet_name,
                        "index": i,
                    }

                    # ID
                    if id_cols:
                        location["id"] = str(row[id_cols[0]])
                    else:
                        location["id"] = f"{sheet_name}_{i}"

                    # Coordinates: skip rows without valid numeric lat/lon
                    if lat_cols and lon_cols:
                        lat_val = row[lat_cols[0]]
                        lon_val = row[lon_cols[0]]
                        try:
                            if pd.notna(lat_val) and pd.notna(lon_val):
                                location["lat"] = float(lat_val)
                                location["lon"] = float(lon_val)
                            else:
                                continue
                        except (TypeError, ValueError):
                            # Something like "GC0" etc. – not a coordinate row
                            continue
                    else:
                        # No coordinate columns at all
                        continue

                    # Name (optional – fall back to ID)
                    location["name"] = str(location.get("id"))

                    # Type – infer from sheet name
                    location["type"] = self._infer_location_type(sheet_name, row)

                    # Add all other columns as attributes
                    for col in df.columns:
                        key = str(col).lower()
                        if col not in (lat_cols + lon_cols + id_cols):
                            location[key] = row[col]

                    locations.append(location)

            logger.info(f"Processed {len(locations)} locations")
            return locations

        except Exception as e:
            logger.error(f"Error processing geo locations: {str(e)}")
            # Return whatever we have instead of nuking everything
            return locations

    
    def _process_capacity_data(self, capacity_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process capacity data"""
        try:
            capacity_records = []

            for sheet_name, df in capacity_data.items():
                logger.info(f"Processing capacity sheet: {sheet_name}")

                df = df.dropna(how="all")
                df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

                for i, row in df.iterrows():
                    base = {
                        "sheet": sheet_name,
                        "index": i,
                    }

                    # Guess ID column
                    id_col = next((col for col in df.columns if 'id' in col or 'name' in col), None)
                    if id_col:
                        base["warehouse_id"] = str(row[id_col])
                    
                    for col in df.columns:
                        if col in ("index", id_col):
                            continue
                        try:
                            value = float(row[col])
                            record = base.copy()
                            record["capacity_type"] = col
                            record["value"] = value
                            capacity_records.append(record)
                        except (ValueError, TypeError):
                            continue

            logger.info(f"Processed {len(capacity_records)} capacity records")
            return capacity_records

        except Exception as e:
            logger.error(f"Error processing capacity data: {str(e)}")
            return []

    
    def _process_demand_data(self, demand_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process demand data"""
        try:
            demand_records = []

            for sheet_name, df in demand_data.items():
                logger.info(f"Processing demand sheet: {sheet_name}")

                # Promote second row to header if first is junk
                if df.shape[0] > 1 and any(str(c).startswith("Unnamed") or str(c).lower() in ["nan", "none"] for c in df.columns):
                    df.columns = df.iloc[0]
                    df = df.drop(df.index[0]).reset_index(drop=True)

                # Drop fully empty or non-numeric columns
                df = df.dropna(how="all", axis=1)
                df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

                # Assume first column is product_type, rest are store IDs
                product_col = df.columns[0]
                store_columns = [c for c in df.columns[1:] if c.isnumeric()]

                for _, row in df.iterrows():
                    product_type = str(row[product_col]).strip().lower()
                    if not product_type or "sum" in product_type or "average" in product_type:
                        continue

                    for store_id in store_columns:
                        try:
                            value = float(row[store_id])
                            demand_records.append({
                                "region": sheet_name,
                                "product_type": product_type,
                                "store_id": str(store_id),
                                "value": value
                            })
                        except (ValueError, TypeError):
                            continue

                logger.info(f"Processed {len(demand_records)} demand records from sheet: {sheet_name}")

            return demand_records

        except Exception as e:
            logger.error(f"Error processing demand data: {str(e)}")
            return []


        
    def _process_distance_data(self, distance_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process multiple distance matrices (regional), store them with names.
        """
        try:
            partial_matrices = {}

            for sheet_name, df in distance_data.items():
                logger.info(f"Processing distance sheet: {sheet_name}")

                df_clean = df.copy()

                # Try to drop columns with Unnamed/nan headers
                df_clean.columns = [str(c) for c in df_clean.columns]
                df_clean = df_clean.loc[:, ~df_clean.columns.str.lower().str.contains("unnamed|nan")]
                df_clean = df_clean.dropna(how="all")

                numeric_df = df_clean.select_dtypes(include=[np.number])

                if not numeric_df.empty and numeric_df.shape[0] > 1:
                    matrix = numeric_df.values.tolist()
                    partial_matrices[sheet_name] = {
                        "rows": numeric_df.shape[0],
                        "cols": numeric_df.shape[1],
                        "matrix": matrix
                    }
                    logger.info(f"Processed partial matrix {sheet_name}: {numeric_df.shape[0]}×{numeric_df.shape[1]}")
                else:
                    logger.warning(f"Sheet {sheet_name} does not contain a usable numeric matrix")

            return {"type": "partial", "matrices": partial_matrices}

        except Exception as e:
            logger.error(f"Error processing distance data: {str(e)}")
            return {"type": "partial", "matrices": {}}


    def _process_time_data(self, time_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process multiple time matrices (regional), store them with names.
        """
        try:
            partial_matrices = {}

            for sheet_name, df in time_data.items():
                logger.info(f"Processing time sheet: {sheet_name}")

                df_clean = df.copy()
                df_clean.columns = [str(c) for c in df_clean.columns]
                df_clean = df_clean.loc[:, ~df_clean.columns.str.lower().str.contains("unnamed|nan")]
                df_clean = df_clean.dropna(how="all")

                numeric_df = df_clean.select_dtypes(include=[np.number])

                if not numeric_df.empty and numeric_df.shape[0] > 1:
                    matrix = numeric_df.values.tolist()
                    partial_matrices[sheet_name] = {
                        "rows": numeric_df.shape[0],
                        "cols": numeric_df.shape[1],
                        "matrix": matrix
                    }
                    logger.info(f"Processed partial time matrix {sheet_name}: {numeric_df.shape[0]}×{numeric_df.shape[1]}")
                else:
                    logger.warning(f"Sheet {sheet_name} does not contain a usable numeric matrix")

            return {"type": "partial", "matrices": partial_matrices}

        except Exception as e:
            logger.error(f"Error processing time data: {str(e)}")
            return {"type": "partial", "matrices": {}}


    
    def _process_cost_data(self, cost_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process standard warehouse-store cost sheets"""
        try:
            cost_records = []

            for sheet_name, df in cost_data.items():
                logger.info(f"Processing cost sheet: {sheet_name}")
                df_clean = df.dropna(how="all").dropna(axis=1, how="all")

                # Try melting if matrix form
                if df_clean.select_dtypes(include=[np.number]).shape[1] > 1:
                    df_melted = df_clean.melt(ignore_index=False).reset_index()
                    df_melted.columns = ['warehouse_id', 'store_id', 'cost']
                    df_melted['sheet'] = sheet_name
                    cost_records.extend(df_melted.to_dict(orient="records"))
                else:
                    # Long-form already
                    for i, row in df_clean.iterrows():
                        record = {"sheet": sheet_name, "index": i}
                        for col in df_clean.columns:
                            record[str(col).lower()] = row[col]
                        cost_records.append(record)

            logger.info(f"Processed {len(cost_records)} cost records")
            return cost_records

        except Exception as e:
            logger.error(f"Error processing cost data: {str(e)}")
            return []
    
    def _process_costs_mwc_data(self, costs_mwc_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process MWC-specific cost sheets"""
        try:
            mwc_records = []

            for sheet_name, df in costs_mwc_data.items():
                logger.info(f"Processing MWC costs sheet: {sheet_name}")
                df_clean = df.dropna(how="all").dropna(axis=1, how="all")

                # Melt if matrix-style
                if df_clean.select_dtypes(include=[np.number]).shape[1] > 1:
                    df_melted = df_clean.melt(ignore_index=False).reset_index()
                    df_melted.columns = ['warehouse_id', 'store_id', 'cost']
                    df_melted['sheet'] = sheet_name
                    mwc_records.extend(df_melted.to_dict(orient="records"))
                else:
                    for i, row in df_clean.iterrows():
                        record = {"sheet": sheet_name, "index": i}
                        for col in df_clean.columns:
                            record[str(col).lower()] = row[col]
                        mwc_records.append(record)

            logger.info(f"Processed {len(mwc_records)} MWC cost records")
            return mwc_records

        except Exception as e:
            logger.error(f"Error processing MWC costs data: {str(e)}")
            return []
    
    def _infer_location_type(self, sheet_name: str, row: pd.Series) -> str:
        """Infer location type from sheet name or row data"""
        sheet_lower = sheet_name.lower()
        row_str = str(row).lower()
        
        if 'warehouse' in sheet_lower or 'warehouse' in row_str:
            return 'warehouse'
        elif 'store' in sheet_lower or 'store' in row_str or 'retail' in row_str:
            return 'store'
        elif 'depot' in sheet_lower or 'depot' in row_str:
            return 'warehouse'
        elif 'customer' in sheet_lower or 'customer' in row_str:
            return 'store'
        else:
            return 'location'
    
    def _separate_warehouses_stores(self, locations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Separate locations into warehouses and stores"""
        warehouses = []
        stores = []
        
        for location in locations:
            location_type = location.get('type', 'location')
            
            if location_type == 'warehouse':
                warehouses.append(location)
            elif location_type == 'store':
                stores.append(location)
            else:
                # If type is unclear, use heuristics
                location_str = str(location).lower()
                if 'warehouse' in location_str or 'depot' in location_str:
                    location['type'] = 'warehouse'
                    warehouses.append(location)
                else:
                    location['type'] = 'store'
                    stores.append(location)
        
        # If we don't have clear separation, split roughly 20/80
        if not warehouses and stores:
            n_warehouses = max(1, len(locations) // 5)
            warehouses = locations[:n_warehouses]
            stores = locations[n_warehouses:]
            
            for w in warehouses:
                w['type'] = 'warehouse'
            for s in stores:
                s['type'] = 'store'
        
        logger.info(f"Separated into {len(warehouses)} warehouses and {len(stores)} stores")
        return warehouses, stores
    
    def _validate_data_consistency(self, processed_data: Dict[str, Any]) -> None:
        """Validate consistency across processed data"""
        try:
            logger.info("Validating data consistency")
            
            # Check if we have minimum required data
            if not processed_data.get('locations'):
                logger.warning("No location data found")
            
            if not processed_data.get('warehouses'):
                logger.warning("No warehouse data found")
            
            if not processed_data.get('stores'):
                logger.warning("No store data found")
            
            # Check matrix dimensions
            distance_matrix = processed_data.get('distance_matrix', [])
            time_matrix = processed_data.get('time_matrix', [])
            n_locations = len(processed_data.get('locations', []))
            
            if distance_matrix and len(distance_matrix) != n_locations:
                logger.warning(f"Distance matrix size ({len(distance_matrix)}) doesn't match number of locations ({n_locations})")
            
            if time_matrix and len(time_matrix) != n_locations:
                logger.warning(f"Time matrix size ({len(time_matrix)}) doesn't match number of locations ({n_locations})")
            
            logger.info("Data validation completed")
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
