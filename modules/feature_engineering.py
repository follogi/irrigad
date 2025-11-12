"""
Feature Engineering Module
Extracts features from sensor data for ML model
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """Extracts and engineers features for ML model"""

    OPTIMAL_MOISTURE = 40.0  # % target
    MIN_IRRIGATION_M3 = 0.1  # Minimum water volume to consider as irrigation (100L)

    def __init__(self):
        self.features_df = None
        self.target_series = None

    def extract_features(self, merged_df: pd.DataFrame, forecast_data: Dict = None) -> pd.DataFrame:
        """
        Extract all features from merged sensor data

        Args:
            merged_df: Merged daily sensor data
            forecast_data: Optional forecast data for prediction

        Returns:
            DataFrame with engineered features
        """
        df = merged_df.copy()
        df['date'] = pd.to_datetime(df['day'])

        # Initialize features dict
        features_list = []

        for i in range(len(df)):
            if i < 14:  # Skip first 14 days (need lookback)
                continue

            row_date = df.iloc[i]['date']
            features = {'date': row_date}

            # 1. SOIL MOISTURE FEATURES
            soil_features = self._extract_soil_moisture_features(df, i)
            features.update(soil_features)

            # 2. WEATHER FEATURES
            weather_features = self._extract_weather_features(df, i)
            features.update(weather_features)

            # 3. IRRIGATION FEATURES
            irrigation_features = self._extract_irrigation_features(df, i)
            features.update(irrigation_features)

            # 4. TEMPORAL FEATURES
            temporal_features = self._extract_temporal_features(row_date)
            features.update(temporal_features)

            # 5. TARGET (for training only)
            if i < len(df) - 1:  # Can calculate target (next day exists)
                target = self._calculate_target(df, i)
                features['target'] = target
            else:
                features['target'] = None

            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        # 6. CALCULATED FEATURES (need weather data)
        if 'precipitation_sum_3d' in features_df.columns and 'temp_mean_3d' in features_df.columns:
            features_df['water_deficit'] = 0.0  # Will be calculated with forecast
            features_df['moisture_deficit'] = self.OPTIMAL_MOISTURE - features_df['avg_soil_moisture']

        self.features_df = features_df
        return features_df

    def _extract_soil_moisture_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Extract soil moisture related features"""
        features = {}

        # Get soil moisture data for last 7 days
        start_idx = max(0, idx - 6)
        sm_window = df.iloc[start_idx:idx + 1]

        # Average of sm1 and sm2
        sm1_vals = sm_window['sm1'].fillna(0).values
        sm2_vals = sm_window['sm2'].fillna(0).values
        avg_sm = (sm1_vals + sm2_vals) / 2.0

        # Features
        features['avg_soil_moisture'] = np.mean(avg_sm[-3:])  # Last 3 days
        features['soil_moisture_std'] = np.std(avg_sm)  # Last 7 days

        # Trend: (today - 3 days ago) / 3
        if len(avg_sm) >= 4:
            features['soil_moisture_trend'] = (avg_sm[-1] - avg_sm[-4]) / 3.0
        else:
            features['soil_moisture_trend'] = 0.0

        return features

    def _extract_weather_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Extract weather related features"""
        features = {}

        # Last 3 days
        window_3d = df.iloc[max(0, idx - 2):idx + 1]

        features['precipitation_sum_3d'] = window_3d['precipitation'].fillna(0).sum()
        features['temp_mean_3d'] = window_3d['airtemperaturemean'].fillna(0).mean()
        features['temp_max_3d'] = window_3d['airtemperaturemean'].fillna(0).max()
        features['humidity_mean_3d'] = window_3d['relativehumiditymean'].fillna(0).mean()
        features['solar_radiation_mean_3d'] = window_3d['solarradiation'].fillna(0).mean()

        # Last 7 days
        window_7d = df.iloc[max(0, idx - 6):idx + 1]
        features['precipitation_sum_7d'] = window_7d['precipitation'].fillna(0).sum()

        return features

    def _extract_irrigation_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Extract irrigation related features"""
        features = {}

        # Calculate daily irrigation from tfm1 (total flow meter)
        tfm1_vals = df['tfm1'].fillna(0).values
        daily_irrigation = np.diff(tfm1_vals)
        daily_irrigation = np.maximum(daily_irrigation, 0)  # Only positive values

        # Last 7 days irrigation
        if idx >= 7:
            features['irrigation_7d'] = np.sum(daily_irrigation[max(0, idx - 7):idx])
        else:
            features['irrigation_7d'] = 0.0

        # Average irrigation last 14 days
        if idx >= 14:
            features['irrigation_avg_14d'] = np.mean(daily_irrigation[max(0, idx - 14):idx])
        else:
            features['irrigation_avg_14d'] = 0.0

        # Days since last significant irrigation (>100L)
        days_since = 0
        for i in range(idx - 1, max(0, idx - 30), -1):  # Look back max 30 days
            if i < len(daily_irrigation) and daily_irrigation[i] > self.MIN_IRRIGATION_M3:
                days_since = idx - i - 1
                break
        features['days_since_last_irrigation'] = days_since

        return features

    def _extract_temporal_features(self, date: datetime) -> Dict[str, Any]:
        """Extract temporal/seasonal features"""
        features = {}

        features['month'] = date.month
        features['day_of_year'] = date.timetuple().tm_yday
        features['is_summer'] = 1 if date.month in [6, 7, 8] else 0

        return features

    def _calculate_target(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calculate target: water irrigated in next 24 hours

        Args:
            df: DataFrame with daily data
            idx: Current index

        Returns:
            Water volume irrigated next day (m³)
        """
        if idx >= len(df) - 1:
            return 0.0

        tfm1_today = df.iloc[idx]['tfm1']
        tfm1_tomorrow = df.iloc[idx + 1]['tfm1']

        # Handle NaN
        if pd.isna(tfm1_today) or pd.isna(tfm1_tomorrow):
            return 0.0

        irrigation = tfm1_tomorrow - tfm1_today

        # Only positive values (counter should be incremental)
        return max(0.0, irrigation)

    def add_forecast_features(self, features: pd.Series, forecast_data: Dict) -> pd.Series:
        """
        Add forecast-based features to a single feature row

        Args:
            features: Series with current features
            forecast_data: Dictionary with forecast data

        Returns:
            Series with added forecast features
        """
        if forecast_data is None or 'daily' not in forecast_data:
            # Default values if no forecast
            features['forecast_precipitation_3d'] = 0.0
            features['forecast_et0_3d'] = 0.0
            features['forecast_vpd_max'] = 0.0
            features['forecast_temp_min'] = 10.0
            features['water_deficit'] = 0.0
            return features

        daily = forecast_data['daily']

        # Sum/aggregate first 3 days
        n_days = min(3, len(daily['time']))

        features['forecast_precipitation_3d'] = sum(daily['precipitation_sum'][:n_days])
        features['forecast_et0_3d'] = sum(daily['et0_fao_evapotranspiration'][:n_days])

        # Max VPD
        if 'vapor_pressure_deficit_max' in daily:
            features['forecast_vpd_max'] = max(daily['vapor_pressure_deficit_max'][:n_days])
        else:
            features['forecast_vpd_max'] = 0.0

        # Min temperature
        features['forecast_temp_min'] = min(daily['temperature_2m_min'][:n_days])

        # Water deficit
        features['water_deficit'] = features['forecast_et0_3d'] - features['forecast_precipitation_3d']

        return features

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data (features and target)

        Returns:
            Tuple of (X: features DataFrame, y: target Series)
        """
        if self.features_df is None:
            raise ValueError("Features not extracted yet. Call extract_features() first.")

        # Remove rows without target
        train_df = self.features_df[self.features_df['target'].notna()].copy()

        # Drop non-feature columns
        feature_cols = [col for col in train_df.columns
                       if col not in ['date', 'target']]

        X = train_df[feature_cols]
        y = train_df['target']

        # Handle any remaining NaN
        X = X.fillna(0)

        return X, y

    def prepare_prediction_features(self, merged_df: pd.DataFrame,
                                   forecast_data: Dict) -> pd.Series:
        """
        Prepare features for current prediction

        Args:
            merged_df: Merged daily sensor data
            forecast_data: Forecast data

        Returns:
            Series with features for prediction
        """
        # Extract features for the last day
        features_df = self.extract_features(merged_df, forecast_data)

        if len(features_df) == 0:
            raise ValueError("Insufficient data to extract features")

        # Get last row (most recent)
        features = features_df.iloc[-1].copy()

        # Add forecast features
        features = self.add_forecast_features(features, forecast_data)

        # Drop non-feature columns
        feature_cols = [col for col in features.index
                       if col not in ['date', 'target']]

        features = features[feature_cols]

        # Handle NaN
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> list:
        """Get list of feature names in correct order"""
        feature_names = [
            'avg_soil_moisture',
            'soil_moisture_std',
            'soil_moisture_trend',
            'precipitation_sum_3d',
            'precipitation_sum_7d',
            'temp_mean_3d',
            'temp_max_3d',
            'humidity_mean_3d',
            'solar_radiation_mean_3d',
            'irrigation_7d',
            'irrigation_avg_14d',
            'days_since_last_irrigation',
            'month',
            'day_of_year',
            'is_summer',
            'forecast_precipitation_3d',
            'forecast_et0_3d',
            'forecast_vpd_max',
            'forecast_temp_min',
            'water_deficit',
            'moisture_deficit'
        ]
        return feature_names

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for each feature"""
        descriptions = {
            'avg_soil_moisture': 'Umidità suolo media',
            'soil_moisture_std': 'Variabilità umidità suolo',
            'soil_moisture_trend': 'Trend umidità',
            'precipitation_sum_3d': 'Pioggia ultimi 3 giorni',
            'precipitation_sum_7d': 'Pioggia ultimi 7 giorni',
            'temp_mean_3d': 'Temperatura media',
            'temp_max_3d': 'Temperatura massima',
            'humidity_mean_3d': 'Umidità relativa',
            'solar_radiation_mean_3d': 'Radiazione solare',
            'irrigation_7d': 'Irrigazione ultimi 7 giorni',
            'irrigation_avg_14d': 'Media irrigazione 14 giorni',
            'days_since_last_irrigation': 'Giorni dall\'ultima irrigazione',
            'month': 'Mese',
            'day_of_year': 'Giorno dell\'anno',
            'is_summer': 'Stagione estiva',
            'forecast_precipitation_3d': 'Pioggia prevista',
            'forecast_et0_3d': 'ET0 prevista',
            'forecast_vpd_max': 'VPD massimo',
            'forecast_temp_min': 'Temperatura minima prevista',
            'water_deficit': 'Deficit idrico',
            'moisture_deficit': 'Deficit umidità suolo'
        }
        return descriptions
