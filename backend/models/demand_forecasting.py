import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

DATA_PATH = r"C:\Users\ganym\OneDrive\Desktop\new\synthetic_demand_dataset.csv"

class DemandForecaster:
    """Demand forecasting using improved Random Forest (internals upgraded)"""

    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=600,
                max_depth=24,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
        }
        self.trained_models = {}
        self.feature_columns = []
        self.target_column = "demand"
        self.encoders = {}

    # --------------------------------------------------
    # INTERNAL FEATURE LOGIC (replaces old logic safely)
    # --------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.sort_values(
                ["node_id", "demand_type", "time_step"]
            ).reset_index(drop=True)

            # Lag features
            for lag in [1, 2, 3, 6, 12]:
                df[f"lag_{lag}"] = df.groupby(
                    ["node_id", "demand_type"]
                )["demand"].shift(lag)

            # Rolling statistics
            df["rolling_mean_3"] = (
                df.groupby(["node_id", "demand_type"])["demand"]
                .shift(1).rolling(3).mean()
            )
            df["rolling_mean_6"] = (
                df.groupby(["node_id", "demand_type"])["demand"]
                .shift(1).rolling(6).mean()
            )
            df["rolling_std_6"] = (
                df.groupby(["node_id", "demand_type"])["demand"]
                .shift(1).rolling(6).std()
            )

            # Trend + seasonality
            df["time_index"] = df["time_step"]
            df["sin_12"] = np.sin(2 * np.pi * df["time_step"] / 12)
            df["cos_12"] = np.cos(2 * np.pi * df["time_step"] / 12)

            df = df.dropna().reset_index(drop=True)

            self.feature_columns = [
                "node_id", "demand_type", "time_index",
                "lag_1","lag_2","lag_3","lag_6","lag_12",
                "rolling_mean_3","rolling_mean_6","rolling_std_6",
                "sin_12","cos_12"
            ]

            return df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    # --------------------------------------------------
    # TRAIN (OUTPUT FORMAT UNCHANGED)
    # --------------------------------------------------
    def train(self, demand_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            logger.info("Starting demand forecasting model training")

            df = pd.read_csv(DATA_PATH)
            df = self.prepare_features(df)

            # Encode categorical columns
            for col in ["node_id", "demand_type"]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le

            # Time-aware split
            train_df = df.groupby(
                ["node_id", "demand_type"]
            ).apply(lambda x: x.iloc[:-12]).reset_index(drop=True)

            test_df = df.groupby(
                ["node_id", "demand_type"]
            ).apply(lambda x: x.iloc[-12:]).reset_index(drop=True)

            X_train = train_df[self.feature_columns]
            y_train = np.log1p(train_df["demand"])

            X_test = test_df[self.feature_columns]
            y_test = test_df["demand"]

            training_results = {}

            for name, model in self.models.items():
                model.fit(X_train, y_train)
                preds = np.expm1(model.predict(X_test))

                training_results[name] = {
                    "train_mae": mean_absolute_error(
                        np.expm1(y_train), np.expm1(model.predict(X_train))
                    ),
                    "test_mae": mean_absolute_error(y_test, preds),
                    "train_rmse": np.sqrt(mean_squared_error(
                        np.expm1(y_train), np.expm1(model.predict(X_train))
                    )),
                    "test_rmse": np.sqrt(mean_squared_error(y_test, preds))
                }

                self.trained_models[name] = model

            logger.info("Demand forecasting training completed")
            return training_results

        except Exception as e:
            logger.error(f"Error in demand forecasting training: {e}")
            raise

    # --------------------------------------------------
    # FORECAST (OUTPUT FORMAT UNCHANGED)
    # --------------------------------------------------
    def forecast(self, demand_data: List[Dict[str, Any]] = None, periods_ahead: int = 12) -> Dict[str, Any]:
        try:
            if not self.trained_models:
                raise ValueError("Models not trained. Call train() first.")

            df = pd.read_csv(DATA_PATH)
            df = self.prepare_features(df)

            forecasts = {}

            for model_name, model in self.trained_models.items():
                model_forecasts = []

                for (nid, dtype), g in df.groupby(["node_id", "demand_type"]):
                    row = g.tail(1).copy()

                    for col in ["node_id", "demand_type"]:
                        row[col] = self.encoders[col].transform(row[col])

                    current = row.copy()

                    series_forecast = []

                    for _ in range(periods_ahead):
                        X = current[self.feature_columns]
                        pred = np.expm1(model.predict(X)[0])
                        series_forecast.append(max(0, pred))

                        # Shift lags correctly
                        current["lag_12"] = current["lag_6"]
                        current["lag_6"] = current["lag_3"]
                        current["lag_3"] = current["lag_2"]
                        current["lag_2"] = current["lag_1"]
                        current["lag_1"] = pred

                    model_forecasts.append(np.mean(series_forecast))

                forecasts[model_name] = model_forecasts

            # Ensemble (same format as old code)
            forecasts["ensemble"] = np.mean(
                list(forecasts.values()), axis=0
            ).tolist()

            return {
                "forecasts_by_model": forecasts,
                "ensemble_forecast": forecasts["ensemble"],
                "total_demand": {
                    "overall": float(df["demand"].mean())
                },
                "forecast_periods": periods_ahead,
                "feature_importance": self._get_feature_importance()
            }

        except Exception as e:
            logger.error(f"Error in demand forecasting: {e}")
            raise

    # --------------------------------------------------
    # FEATURE IMPORTANCE (UNCHANGED)
    # --------------------------------------------------
    def _get_feature_importance(self) -> Dict[str, Any]:
        try:
            importance = {}
            if "random_forest" in self.trained_models:
                importance["random_forest"] = dict(
                    zip(
                        self.feature_columns,
                        self.trained_models["random_forest"].feature_importances_
                    )
                )
            return importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
