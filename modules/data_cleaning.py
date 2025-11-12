"""
Data Cleaning Module for Bluetentacles AI
Automatically cleans outliers and anomalies in uploaded CSV files
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Pulisce dati CSV da outliers e anomalie
    """

    def __init__(self):
        self.cleaning_report = {}

    def clean_all_data(
        self,
        valve_df: pd.DataFrame,
        sm_df: pd.DataFrame,
        meteo_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Pulisce tutti i dataframe

        Args:
            valve_df: DataFrame con dati irrigazione
            sm_df: DataFrame con dati umidità suolo
            meteo_df: DataFrame con dati meteo

        Returns:
            (valve_clean, sm_clean, meteo_clean, report)
        """
        logger.info("Inizio pulizia dati...")

        # 1. Pulisci valve (irrigazione) - PRIORITÀ MASSIMA
        valve_clean = self.clean_valve_data(valve_df)

        # 2. Pulisci sm (umidità suolo)
        sm_clean = self.clean_soil_moisture_data(sm_df)

        # 3. Pulisci meteo
        meteo_clean = self.clean_meteo_data(meteo_df)

        # 4. Genera report
        report = self.generate_cleaning_report()

        logger.info(f"Pulizia completata: {report['summary']['total_outliers']} outliers rimossi")

        return valve_clean, sm_clean, meteo_clean, report

    def clean_valve_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pulisce dati valve.csv da outliers irrigazione

        CRITICI DA RISOLVERE:
        1. Irrigazione negativa (reset contatore)
        2. Irrigazione > 5000 m³/giorno (anomalo)
        3. Spike improvvisi non realistici
        """
        logger.info(f"Pulizia valve.csv: {len(df)} records")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Estrai tfm1 (contatore totale acqua)
        tfm1_mask = df['name'] == 'tfm1'

        if tfm1_mask.sum() == 0:
            logger.warning("Nessun dato tfm1 trovato in valve.csv")
            self.cleaning_report['valve'] = {
                'outliers_found': 0,
                'outlier_days': [],
                'negative_values': 0,
                'too_high_values': 0,
                'iqr_outliers': 0,
                'stats_before': {},
                'stats_after': {},
                'variance_reduction_pct': 0.0,
                'interpolation_method': 'none',
                'fallback_method': 'none'
            }
            return df

        tfm1_data = df[tfm1_mask].copy()
        tfm1_data = tfm1_data.sort_values('date')

        # Calcola irrigazione giornaliera
        tfm1_data['day'] = tfm1_data['date'].dt.date
        tfm1_daily = tfm1_data.groupby('day').agg({
            'value': ['first', 'last']
        })
        tfm1_daily.columns = ['first', 'last']
        tfm1_daily['irrigation'] = tfm1_daily['last'] - tfm1_daily['first']

        # STATISTICHE PRE-PULIZIA
        stats_before = {
            'min': float(tfm1_daily['irrigation'].min()),
            'max': float(tfm1_daily['irrigation'].max()),
            'mean': float(tfm1_daily['irrigation'].mean()),
            'std': float(tfm1_daily['irrigation'].std()),
            'variance': float(tfm1_daily['irrigation'].var())
        }

        logger.info(f"Stats pre-pulizia: min={stats_before['min']:.1f}, max={stats_before['max']:.1f}, mean={stats_before['mean']:.1f}, std={stats_before['std']:.1f}")

        # IDENTIFICA OUTLIERS
        negative_mask = tfm1_daily['irrigation'] < 0
        too_high_mask = tfm1_daily['irrigation'] > 5000

        # Calcola outliers usando IQR (Interquartile Range)
        Q1 = tfm1_daily['irrigation'].quantile(0.25)
        Q3 = tfm1_daily['irrigation'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3×IQR per outliers estremi
        upper_bound = Q3 + 3 * IQR

        iqr_outliers = (tfm1_daily['irrigation'] < lower_bound) | \
                       (tfm1_daily['irrigation'] > upper_bound)

        # Combina tutti gli outliers
        outliers_mask = negative_mask | too_high_mask | iqr_outliers

        n_outliers = int(outliers_mask.sum())
        outlier_days = [str(d) for d in tfm1_daily[outliers_mask].index.tolist()]

        logger.info(f"Outliers trovati: {n_outliers} giorni ({n_outliers/len(tfm1_daily)*100:.1f}%)")
        logger.info(f"  - Negativi: {int(negative_mask.sum())}")
        logger.info(f"  - Troppo alti (>5000): {int(too_high_mask.sum())}")
        logger.info(f"  - IQR outliers: {int(iqr_outliers.sum())}")

        if n_outliers > 0:
            logger.debug(f"Giorni outlier: {outlier_days[:10]}...")  # Primi 10

        # CORREZIONE OUTLIERS
        tfm1_daily['irrigation_clean'] = tfm1_daily['irrigation'].copy()

        # Sostituisci outliers con NaN
        tfm1_daily.loc[outliers_mask, 'irrigation_clean'] = np.nan

        # INTERPOLAZIONE
        # Usa interpolazione lineare per riempire i buchi
        tfm1_daily['irrigation_clean'] = tfm1_daily['irrigation_clean'].interpolate(
            method='linear',
            limit_direction='both',
            limit=5  # Max 5 giorni consecutivi interpolati
        )

        # Se ancora NaN (es. inizio/fine dataset), usa mediana
        median_irrigation = tfm1_daily['irrigation_clean'].median()
        tfm1_daily['irrigation_clean'] = tfm1_daily['irrigation_clean'].fillna(median_irrigation)

        # Assicurati che non ci siano valori negativi dopo interpolazione
        tfm1_daily['irrigation_clean'] = tfm1_daily['irrigation_clean'].clip(lower=0)

        # RICOSTRUISCI CONTATORE CUMULATIVO
        # Calcola nuovo tfm1 cumulativo senza outliers
        first_value = tfm1_data['value'].iloc[0]
        tfm1_daily['tfm1_cumulative'] = tfm1_daily['irrigation_clean'].cumsum() + first_value

        # AGGIORNA DATAFRAME ORIGINALE
        # Per ogni giorno, interpola linearmente i valori tfm1 durante il giorno
        for day_date, day_row in tfm1_daily.iterrows():
            day_mask = tfm1_mask & (df['date'].dt.date == day_date)

            if day_mask.sum() > 0:
                day_records = df[day_mask].sort_values('date')
                n_records = len(day_records)

                # Calcola valori interpolati per il giorno
                if day_date == tfm1_daily.index[0]:
                    start_val = first_value
                else:
                    prev_idx = tfm1_daily.index.get_loc(day_date) - 1
                    start_val = tfm1_daily.iloc[prev_idx]['tfm1_cumulative']

                end_val = day_row['tfm1_cumulative']

                interpolated_values = np.linspace(start_val, end_val, n_records)
                df.loc[day_mask, 'value'] = interpolated_values

        # STATISTICHE POST-PULIZIA
        stats_after = {
            'min': float(tfm1_daily['irrigation_clean'].min()),
            'max': float(tfm1_daily['irrigation_clean'].max()),
            'mean': float(tfm1_daily['irrigation_clean'].mean()),
            'std': float(tfm1_daily['irrigation_clean'].std()),
            'variance': float(tfm1_daily['irrigation_clean'].var())
        }

        logger.info(f"Stats post-pulizia: min={stats_after['min']:.1f}, max={stats_after['max']:.1f}, mean={stats_after['mean']:.1f}, std={stats_after['std']:.1f}")

        # SALVA REPORT
        if stats_before['variance'] > 0:
            variance_reduction = (1 - stats_after['variance'] / stats_before['variance']) * 100
        else:
            variance_reduction = 0.0

        logger.info(f"Varianza ridotta: {variance_reduction:.1f}%")

        self.cleaning_report['valve'] = {
            'outliers_found': n_outliers,
            'outlier_days': outlier_days,
            'negative_values': int(negative_mask.sum()),
            'too_high_values': int(too_high_mask.sum()),
            'iqr_outliers': int(iqr_outliers.sum()),
            'stats_before': stats_before,
            'stats_after': stats_after,
            'variance_reduction_pct': float(variance_reduction),
            'interpolation_method': 'linear',
            'fallback_method': 'median'
        }

        return df

    def clean_soil_moisture_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pulisce dati sm.csv (umidità suolo)

        PROBLEMI DA RISOLVERE:
        1. Valori < 0% o > 100% (fisicamente impossibili)
        2. Spike improvvisi (es: da 30% a 90% in 1 ora)
        3. Valori costanti prolungati (sensore bloccato)
        """
        logger.info(f"Pulizia sm.csv: {len(df)} records")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        outliers_removed = 0
        sensors_cleaned = []

        for sensor in ['sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'sm6']:
            sensor_mask = df['name'] == sensor

            if sensor_mask.sum() == 0:
                continue

            sensors_cleaned.append(sensor)
            sensor_data = df[sensor_mask].copy()
            sensor_data = sensor_data.sort_values('date')

            # IDENTIFICA OUTLIERS

            # 1. Valori fuori range [0, 100]
            invalid_range = (sensor_data['value'] < 0) | (sensor_data['value'] > 100)

            # 2. Spike improvvisi (cambio >30% in 1 ora)
            sensor_data['diff'] = sensor_data['value'].diff().abs()
            spikes = sensor_data['diff'] > 30

            # 3. Valori costanti >24h (sensore bloccato)
            sensor_data['is_constant'] = (sensor_data['value'].diff() == 0)
            # Rolling window per trovare 24+ ore consecutive costanti
            min_constant_readings = 24  # assumendo letture orarie
            sensor_data['constant_streak'] = sensor_data.groupby(
                (sensor_data['is_constant'] != sensor_data['is_constant'].shift()).cumsum()
            )['is_constant'].transform('sum')
            stuck_sensor = (sensor_data['constant_streak'] > min_constant_readings) & sensor_data['is_constant']

            # Combina outliers
            outliers = invalid_range | spikes | stuck_sensor
            n_outliers = int(outliers.sum())
            outliers_removed += n_outliers

            if n_outliers > 0:
                logger.info(f"  {sensor}: rimossi {n_outliers} outliers")

                # Sostituisci con interpolazione
                sensor_data.loc[outliers, 'value'] = np.nan
                sensor_data['value'] = sensor_data['value'].interpolate(
                    method='linear',
                    limit_direction='both'
                )

                # Clip al range valido
                sensor_data['value'] = sensor_data['value'].clip(lower=0, upper=100)

                # Aggiorna dataframe principale
                df.loc[sensor_mask, 'value'] = sensor_data['value'].values

        logger.info(f"SM pulizia completata: {outliers_removed} outliers totali rimossi da {len(sensors_cleaned)} sensori")

        self.cleaning_report['sm'] = {
            'outliers_removed': outliers_removed,
            'sensors_checked': sensors_cleaned
        }

        return df

    def clean_meteo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pulisce dati meteo.csv

        PROBLEMI DA RISOLVERE:
        1. Temperature fuori range (-50°C a +60°C)
        2. Precipitazioni negative
        3. Umidità relativa < 0% o > 100%
        4. Radiazione solare negativa
        """
        logger.info(f"Pulizia meteo.csv: {len(df)} records")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        outliers_by_sensor = {}

        # REGOLE DI VALIDAZIONE
        validation_rules = {
            'airtemperaturemean': (-50, 60),
            'airtemperaturemin': (-50, 60),
            'airtemperaturemax': (-50, 60),
            'precipitation': (0, 500),  # max 500mm/giorno
            'relativehumiditymean': (0, 100),
            'relativehumiditymin': (0, 100),
            'relativehumiditymax': (0, 100),
            'solarradiation': (0, 1500),  # max 1500 W/m²
            'windspeedmean': (0, 50),  # max 50 m/s
            'windspeedmax': (0, 80),
            'soiltemperaturemean': (-20, 60),
            'soiltemperaturemin': (-20, 60),
            'soiltemperaturemax': (-20, 60)
        }

        for sensor, (min_val, max_val) in validation_rules.items():
            sensor_mask = df['name'] == sensor

            if sensor_mask.sum() == 0:
                continue

            # Identifica outliers
            invalid_values = (df.loc[sensor_mask, 'value'] < min_val) | \
                           (df.loc[sensor_mask, 'value'] > max_val)

            n_outliers = int(invalid_values.sum())

            if n_outliers > 0:
                logger.info(f"  {sensor}: rimossi {n_outliers} outliers (range: [{min_val}, {max_val}])")

                # Sostituisci con interpolazione
                sensor_data = df.loc[sensor_mask, 'value'].copy()
                sensor_data[invalid_values] = np.nan
                sensor_data = sensor_data.interpolate(method='linear', limit_direction='both')

                # Clip al range valido
                sensor_data = sensor_data.clip(lower=min_val, upper=max_val)

                df.loc[sensor_mask, 'value'] = sensor_data

                outliers_by_sensor[sensor] = n_outliers

        total_outliers = sum(outliers_by_sensor.values())
        logger.info(f"Meteo pulizia completata: {total_outliers} outliers totali rimossi")

        self.cleaning_report['meteo'] = {
            'outliers_by_sensor': outliers_by_sensor,
            'total_outliers': total_outliers
        }

        return df

    def generate_cleaning_report(self) -> Dict:
        """
        Genera report leggibile della pulizia
        """
        valve_outliers = self.cleaning_report.get('valve', {}).get('outliers_found', 0)
        sm_outliers = self.cleaning_report.get('sm', {}).get('outliers_removed', 0)
        meteo_outliers = self.cleaning_report.get('meteo', {}).get('total_outliers', 0)

        total_outliers = valve_outliers + sm_outliers + meteo_outliers

        report = {
            'summary': {
                'valve_outliers': valve_outliers,
                'sm_outliers': sm_outliers,
                'meteo_outliers': meteo_outliers,
                'total_outliers': total_outliers
            },
            'details': self.cleaning_report
        }

        return report
