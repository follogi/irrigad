"""
Data Loader Module
Handles CSV/JSON parsing and validation
"""

import pandas as pd
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np


class DataLoader:
    """Loads and validates sensor data files"""

    ERROR_MESSAGES = {
        'file_missing': '❌ File {filename} mancante. Carica tutti i 4 file richiesti.',
        'invalid_format': '❌ Formato file {filename} non valido. Controlla intestazioni CSV.',
        'period_mismatch': '⚠️ I periodi dei file non coincidono. sm.csv, meteo.csv e valve.csv devono coprire lo stesso intervallo temporale.',
        'insufficient_data': '❌ Dati insufficienti per training ML (minimo 60 giorni). Caricati solo {days} giorni.',
        'ml_training_failed': '⚠️ Training ML fallito (R² troppo basso). Uso algoritmo regole fisiche.',
        'no_irrigation_data': '⚠️ Nessuna irrigazione rilevata nel periodo. Verifica che tfm1 abbia valori incrementali.',
        'missing_sensors': '❌ Sensori essenziali mancanti in {filename}: {sensors}',
        'invalid_json': '❌ File JSON non valido: {error}'
    }

    def __init__(self):
        self.sm_df = None
        self.meteo_df = None
        self.valve_df = None
        self.forecast_data = None

    def load_csv(self, filepath: str, file_type: str) -> Tuple[bool, List[str]]:
        """
        Load and validate CSV file

        Args:
            filepath: Path to CSV file
            file_type: Type of file ('sm', 'meteo', 'valve')

        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []

        try:
            df = pd.read_csv(filepath)

            # Validate required columns
            required_cols = ['id_device', 'name', 'date', 'value']
            if not all(col in df.columns for col in required_cols):
                errors.append(self.ERROR_MESSAGES['invalid_format'].format(filename=f"{file_type}.csv"))
                return False, errors

            # Convert date column
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                errors.append(f"❌ Errore parsing date in {file_type}.csv: {str(e)}")
                return False, errors

            # Sort by date
            df = df.sort_values('date')

            # Type-specific validations
            validation_result = self._validate_file_type(df, file_type)
            if not validation_result[0]:
                errors.extend(validation_result[1])
                return False, errors

            # Store dataframe
            if file_type == 'sm':
                self.sm_df = df
            elif file_type == 'meteo':
                self.meteo_df = df
            elif file_type == 'valve':
                self.valve_df = df

            return True, []

        except Exception as e:
            errors.append(f"❌ Errore caricamento {file_type}.csv: {str(e)}")
            return False, errors

    def _validate_file_type(self, df: pd.DataFrame, file_type: str) -> Tuple[bool, List[str]]:
        """Validate specific file type requirements"""
        errors = []
        sensors = df['name'].unique()

        if file_type == 'sm':
            required_sensors = ['sm1', 'sm2']
            missing = [s for s in required_sensors if s not in sensors]
            if missing:
                errors.append(self.ERROR_MESSAGES['missing_sensors'].format(
                    filename='sm.csv',
                    sensors=', '.join(missing)
                ))

        elif file_type == 'meteo':
            required_sensors = ['precipitation', 'airtemperaturemean', 'relativehumiditymean']
            missing = [s for s in required_sensors if s not in sensors]
            if missing:
                errors.append(self.ERROR_MESSAGES['missing_sensors'].format(
                    filename='meteo.csv',
                    sensors=', '.join(missing)
                ))

        elif file_type == 'valve':
            if 'tfm1' not in sensors:
                errors.append(self.ERROR_MESSAGES['missing_sensors'].format(
                    filename='valve.csv',
                    sensors='tfm1'
                ))

        # Check minimum period (60 days)
        days = (df['date'].max() - df['date'].min()).days
        if days < 60:
            errors.append(self.ERROR_MESSAGES['insufficient_data'].format(days=days))

        return len(errors) == 0, errors

    def load_json(self, filepath: str) -> Tuple[bool, List[str]]:
        """
        Load and validate forecast JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Validate structure
            if 'daily' not in data:
                errors.append(self.ERROR_MESSAGES['invalid_json'].format(
                    error="Missing 'daily' key"
                ))
                return False, errors

            daily = data['daily']
            required_keys = ['time', 'temperature_2m_max', 'temperature_2m_min',
                           'precipitation_sum', 'et0_fao_evapotranspiration']

            missing = [k for k in required_keys if k not in daily]
            if missing:
                errors.append(self.ERROR_MESSAGES['invalid_json'].format(
                    error=f"Missing keys: {', '.join(missing)}"
                ))
                return False, errors

            # Check forecast period (at least 3 days)
            if len(daily['time']) < 3:
                errors.append("⚠️ Previsioni meteo devono coprire almeno 3 giorni")
                return False, errors

            self.forecast_data = data
            return True, []

        except json.JSONDecodeError as e:
            errors.append(self.ERROR_MESSAGES['invalid_json'].format(error=str(e)))
            return False, errors
        except Exception as e:
            errors.append(f"❌ Errore caricamento previsioni.json: {str(e)}")
            return False, errors

    def validate_periods(self) -> Tuple[bool, List[str]]:
        """
        Validate that all CSV files cover overlapping periods

        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []

        if self.sm_df is None or self.meteo_df is None or self.valve_df is None:
            errors.append("❌ Non tutti i file CSV sono stati caricati")
            return False, errors

        # Get date ranges
        sm_range = (self.sm_df['date'].min(), self.sm_df['date'].max())
        meteo_range = (self.meteo_df['date'].min(), self.meteo_df['date'].max())
        valve_range = (self.valve_df['date'].min(), self.valve_df['date'].max())

        # Check overlap
        start = max(sm_range[0], meteo_range[0], valve_range[0])
        end = min(sm_range[1], meteo_range[1], valve_range[1])

        if start >= end:
            errors.append(self.ERROR_MESSAGES['period_mismatch'])
            return False, errors

        overlap_days = (end - start).days
        if overlap_days < 60:
            errors.append(self.ERROR_MESSAGES['insufficient_data'].format(days=overlap_days))
            return False, errors

        return True, []

    def get_merged_data(self) -> pd.DataFrame:
        """
        Merge all CSV data into a single DataFrame with daily aggregations

        Returns:
            DataFrame with daily aggregated sensor data
        """
        if self.sm_df is None or self.meteo_df is None or self.valve_df is None:
            raise ValueError("Not all CSV files loaded")

        # Create date column (date only, no time)
        self.sm_df['day'] = self.sm_df['date'].dt.date
        self.meteo_df['day'] = self.meteo_df['date'].dt.date
        self.valve_df['day'] = self.valve_df['date'].dt.date

        # Pivot and aggregate each dataset
        sm_pivot = self._pivot_sensors(self.sm_df, ['sm1', 'sm2'])
        meteo_pivot = self._pivot_sensors(self.meteo_df, [
            'precipitation', 'airtemperaturemean', 'airtemperaturemin', 'airtemperaturemax',
            'relativehumiditymean', 'solarradiation', 'windspeedmean', 'soiltemperaturemean'
        ])
        valve_pivot = self._pivot_sensors(self.valve_df, ['tfm1', 'valve1', 'valve2', 'valve3', 'valve4'])

        # Merge all data on day
        merged = sm_pivot.merge(meteo_pivot, on='day', how='inner')
        merged = merged.merge(valve_pivot, on='day', how='inner')

        # Sort by date
        merged = merged.sort_values('day').reset_index(drop=True)

        return merged

    def _pivot_sensors(self, df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
        """
        Pivot sensor data to wide format with daily aggregation

        Args:
            df: DataFrame with sensor data
            sensors: List of sensor names to include

        Returns:
            Pivoted DataFrame with one row per day
        """
        # Filter to relevant sensors
        df_filtered = df[df['name'].isin(sensors)].copy()

        # Group by day and sensor, take mean for numeric sensors
        daily = df_filtered.groupby(['day', 'name'])['value'].mean().reset_index()

        # Pivot to wide format
        pivot = daily.pivot(index='day', columns='name', values='value').reset_index()

        # Fill missing columns with NaN
        for sensor in sensors:
            if sensor not in pivot.columns:
                pivot[sensor] = np.nan

        return pivot

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded data

        Returns:
            Dictionary with summary info
        """
        summary = {}

        if self.sm_df is not None:
            summary['sm'] = {
                'start_date': self.sm_df['date'].min().strftime('%Y-%m-%d'),
                'end_date': self.sm_df['date'].max().strftime('%Y-%m-%d'),
                'days': (self.sm_df['date'].max() - self.sm_df['date'].min()).days,
                'sensors': self.sm_df['name'].unique().tolist()
            }

        if self.meteo_df is not None:
            summary['meteo'] = {
                'start_date': self.meteo_df['date'].min().strftime('%Y-%m-%d'),
                'end_date': self.meteo_df['date'].max().strftime('%Y-%m-%d'),
                'days': (self.meteo_df['date'].max() - self.meteo_df['date'].min()).days,
                'sensors': self.meteo_df['name'].unique().tolist()
            }

        if self.valve_df is not None:
            summary['valve'] = {
                'start_date': self.valve_df['date'].min().strftime('%Y-%m-%d'),
                'end_date': self.valve_df['date'].max().strftime('%Y-%m-%d'),
                'days': (self.valve_df['date'].max() - self.valve_df['date'].min()).days,
                'sensors': self.valve_df['name'].unique().tolist()
            }

        if self.forecast_data is not None:
            summary['forecast'] = {
                'start_date': self.forecast_data['daily']['time'][0],
                'end_date': self.forecast_data['daily']['time'][-1],
                'days': len(self.forecast_data['daily']['time'])
            }

        return summary
