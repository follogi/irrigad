"""
Rule-Based Algorithm Module
Fallback algorithm based on FAO-56 agronomic rules
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class RuleBasedPredictor:
    """
    Rule-based irrigation prediction using agronomic principles
    Based on FAO-56 guidelines
    """

    # Agronomic constants
    OPTIMAL_MOISTURE = 40.0  # % target soil moisture
    MIN_MOISTURE = 25.0      # % stress threshold
    MAX_DAILY_IRRIGATION = 15.0  # mm/day maximum limit
    FIELD_AREA_HA = 10.0     # Default field area in hectares (for m³ to mm conversion)

    def __init__(self):
        self.last_prediction = None

    def predict(self, features: pd.Series, forecast_data: Dict) -> Dict[str, Any]:
        """
        Calculate irrigation recommendation using rule-based algorithm

        Args:
            features: Feature values (from FeatureEngineer)
            forecast_data: Weather forecast data

        Returns:
            Dictionary with prediction and explanation
        """
        # Extract key values
        current_moisture = features.get('avg_soil_moisture', 30.0)
        precipitation_3d = features.get('precipitation_sum_3d', 0.0)
        soil_trend = features.get('soil_moisture_trend', 0.0)
        days_since_irrigation = features.get('days_since_last_irrigation', 0)

        # Forecast values
        et0_3d = features.get('forecast_et0_3d', 0.0)
        rain_3d = features.get('forecast_precipitation_3d', 0.0)
        vpd_max = features.get('forecast_vpd_max', 0.0)
        temp_min = features.get('forecast_temp_min', 10.0)

        # 1. Calculate base water deficit
        water_deficit_base = et0_3d - rain_3d

        # 2. Moisture correction factor
        moisture_factor = self._calculate_moisture_factor(current_moisture)

        # 3. Climate adjustment factor
        climate_factor, frost_warning = self._calculate_climate_factor(
            precipitation_3d, vpd_max, temp_min, soil_trend
        )

        # 4. Calculate water requirement
        water_base = water_deficit_base * moisture_factor
        water_adjusted = water_base * climate_factor

        # 5. Recovery water for deficit
        if current_moisture < 30:
            recovery = (self.OPTIMAL_MOISTURE - current_moisture) * 0.5
            water_adjusted += recovery

        # 6. Apply safety limits
        water_mm = max(0, min(water_adjusted, self.MAX_DAILY_IRRIGATION))

        # 7. Convert to m³ (assuming 10 ha field = 100,000 m²)
        water_m3 = water_mm * self.FIELD_AREA_HA * 100  # 1mm over 1ha = 10m³

        # 8. Determine confidence level
        confidence = self._calculate_confidence(
            frost_warning, current_moisture, days_since_irrigation
        )

        # 9. Determine priority
        priority = self._calculate_priority(current_moisture, water_deficit_base)

        # 10. Generate explanation
        explanation = self._generate_explanation(
            water_mm=water_mm,
            current_moisture=current_moisture,
            water_deficit_base=water_deficit_base,
            moisture_factor=moisture_factor,
            climate_factor=climate_factor,
            frost_warning=frost_warning,
            vpd_max=vpd_max,
            soil_trend=soil_trend
        )

        result = {
            'water_m3': water_m3,
            'water_mm': water_mm,
            'confidence': confidence,
            'priority': priority,
            'frost_warning': frost_warning,
            'explanation': explanation,
            'factors': {
                'water_deficit_base': water_deficit_base,
                'moisture_factor': moisture_factor,
                'climate_factor': climate_factor,
                'current_moisture': current_moisture
            }
        }

        self.last_prediction = result
        return result

    def _calculate_moisture_factor(self, current_moisture: float) -> float:
        """
        Calculate multiplication factor based on current soil moisture

        Args:
            current_moisture: Current soil moisture (%)

        Returns:
            Multiplication factor
        """
        if current_moisture < self.MIN_MOISTURE:
            return 1.8  # Severe stress
        elif current_moisture < 30:
            return 1.5  # Below target
        elif current_moisture < 35:
            return 1.2
        elif current_moisture < self.OPTIMAL_MOISTURE:
            return 1.0
        elif current_moisture < 45:
            return 0.5  # Already good
        else:
            return 0.0  # Saturated

    def _calculate_climate_factor(self, precipitation_3d: float, vpd_max: float,
                                  temp_min: float, soil_trend: float) -> tuple:
        """
        Calculate climate adjustment factor

        Args:
            precipitation_3d: Recent precipitation (mm)
            vpd_max: Maximum vapor pressure deficit (kPa)
            temp_min: Minimum temperature forecast (°C)
            soil_trend: Soil moisture trend

        Returns:
            Tuple of (factor, frost_warning)
        """
        factor = 1.0
        frost_warning = False

        # Recent precipitation adjustment
        if precipitation_3d > 10:
            factor *= 0.7
        elif precipitation_3d > 5:
            factor *= 0.85

        # VPD adjustment (evaporative demand)
        if vpd_max > 2.0:
            factor *= 1.3  # High evaporative demand
        elif vpd_max > 1.5:
            factor *= 1.15

        # Frost risk
        if temp_min < 2:
            factor *= 0.5  # Reduce irrigation
            frost_warning = True

        # Soil moisture trend
        if soil_trend < -0.5:
            factor *= 1.1  # Rapid drying

        return factor, frost_warning

    def _calculate_confidence(self, frost_warning: bool, current_moisture: float,
                            days_since_irrigation: int) -> str:
        """
        Calculate confidence level of prediction

        Args:
            frost_warning: Frost risk flag
            current_moisture: Current soil moisture (%)
            days_since_irrigation: Days since last irrigation

        Returns:
            Confidence level: 'ALTA', 'MEDIA', 'BASSA'
        """
        if frost_warning:
            return "BASSA"
        elif current_moisture < self.MIN_MOISTURE:
            return "MEDIA"
        elif days_since_irrigation > 7:
            return "MEDIA"
        else:
            return "ALTA"

    def _calculate_priority(self, current_moisture: float,
                          water_deficit: float) -> str:
        """
        Calculate irrigation priority

        Args:
            current_moisture: Current soil moisture (%)
            water_deficit: Water deficit (mm)

        Returns:
            Priority level: 'ALTA', 'MEDIA', 'BASSA'
        """
        if current_moisture < self.MIN_MOISTURE:
            return "ALTA"
        elif current_moisture < 30 and water_deficit > 3:
            return "ALTA"
        elif current_moisture < 35:
            return "MEDIA"
        else:
            return "BASSA"

    def _generate_explanation(self, water_mm: float, current_moisture: float,
                            water_deficit_base: float, moisture_factor: float,
                            climate_factor: float, frost_warning: bool,
                            vpd_max: float, soil_trend: float) -> Dict[str, Any]:
        """
        Generate human-readable explanation of the decision

        Args:
            Various calculation parameters

        Returns:
            Dictionary with explanation components
        """
        # Summary
        if frost_warning:
            summary = (f"Raccomandazione ridotta per rischio gelo. "
                      f"Irrigazione: {water_mm:.1f}mm. "
                      f"Attenzione: irrigare nelle ore calde (12:00-16:00).")
        elif current_moisture < self.MIN_MOISTURE:
            summary = (f"URGENTE: umidità suolo sotto soglia stress ({current_moisture:.1f}%). "
                      f"Irrigazione immediata: {water_mm:.1f}mm.")
        else:
            summary = (f"Raccomandazione basata su regole FAO-56. "
                      f"Irrigazione: {water_mm:.1f}mm per mantenere umidità ottimale.")

        # Decision steps
        steps = [
            f"1. Deficit idrico base: {water_deficit_base:.2f}mm (ET0 - Pioggia prevista)",
            f"2. Fattore correzione umidità: {moisture_factor:.2f}x (umidità attuale: {current_moisture:.1f}%)",
            f"3. Base calcolata: {water_deficit_base:.2f} × {moisture_factor:.2f} = {water_deficit_base * moisture_factor:.2f}mm",
            f"4. Fattore climatico: {climate_factor:.2f}x (VPD: {vpd_max:.2f}kPa, trend: {soil_trend:.2f})",
            f"5. TOTALE RACCOMANDATO: {water_mm:.1f}mm"
        ]

        # Key factors
        key_factors = [
            {
                "factor": "Umidità suolo attuale",
                "value": f"{current_moisture:.1f}%",
                "impact": "Alto" if abs(current_moisture - self.OPTIMAL_MOISTURE) > 5 else "Medio",
                "weight": "35%",
                "note": f"Target: {self.OPTIMAL_MOISTURE}%"
            },
            {
                "factor": "Deficit idrico previsto",
                "value": f"{water_deficit_base:.2f}mm",
                "impact": "Alto" if water_deficit_base > 5 else "Medio",
                "weight": "30%",
                "note": "ET0 - Precipitazioni"
            },
            {
                "factor": "VPD (domanda evapotraspirativa)",
                "value": f"{vpd_max:.2f}kPa",
                "impact": "Alto" if vpd_max > 1.5 else "Basso",
                "weight": "20%",
                "note": "Stress pianta"
            },
            {
                "factor": "Trend umidità",
                "value": f"{soil_trend:.2f}%/giorno",
                "impact": "Medio",
                "weight": "15%",
                "note": "Velocità essiccamento"
            }
        ]

        # Warnings
        warnings = []
        if frost_warning:
            warnings.append("⚠️ ATTENZIONE: Rischio gelo - temperatura minima < 2°C")
            warnings.append("💡 Irrigare preferibilmente tra le 12:00 e le 16:00")
        if current_moisture < self.MIN_MOISTURE:
            warnings.append("🚨 URGENTE: Piante in stress idrico!")
        if current_moisture > 45:
            warnings.append("ℹ️ Umidità suolo già elevata - ridurre irrigazione")

        warnings.append("📊 Monitorare umidità suolo nelle prossime 24h")

        return {
            'summary': summary,
            'steps': steps,
            'key_factors': key_factors,
            'warnings': warnings
        }

    def get_method_info(self) -> Dict[str, str]:
        """Get information about the rule-based method"""
        return {
            'name': 'Rule-Based FAO-56',
            'description': 'Algoritmo basato su regole agronomiche FAO-56',
            'confidence': 'Media',
            'note': 'Utilizzato come fallback quando ML non disponibile o poco affidabile'
        }
