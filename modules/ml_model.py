"""
Machine Learning Model Module
RandomForest model for irrigation prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import os


class MLPredictor:
    """
    Machine Learning predictor using RandomForest
    """

    MIN_R2_SCORE = 0.5  # Minimum R² to use ML model (otherwise fallback to rules)
    FIELD_AREA_HA = 10.0  # Default field area for mm to m³ conversion

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = None
        self.scaler = StandardScaler()  # Feature scaling per equalizzare le scale

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train RandomForest model

        Args:
            X: Feature matrix
            y: Target vector (irrigation m³)

        Returns:
            Dictionary with training metrics
        """
        if len(X) < 30:
            raise ValueError(f"Insufficient training data: {len(X)} samples (minimum 30)")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ========== FEATURE SCALING ==========
        # Scala features per equalizzare le scale (irrigation_7d: 0-17k vs vpd: 0-1.5)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Log effetto scaling
        print(f"\n{'='*60}")
        print(f"📊 FEATURE SCALING APPLICATO:")

        # Trova indice di irrigation_7d (feature con valore grande)
        if 'irrigation_7d' in self.feature_names:
            irr_idx = self.feature_names.index('irrigation_7d')
            print(f"   irrigation_7d PRIMA:  max = {X_train.iloc[:, irr_idx].max():.0f} m³")
            print(f"   irrigation_7d DOPO:   max = {X_train_scaled[:, irr_idx].max():.2f} (normalizzato)")

        # Trova indice di forecast_et0_3d (feature con valore piccolo)
        if 'forecast_et0_3d' in self.feature_names:
            et0_idx = self.feature_names.index('forecast_et0_3d')
            print(f"   forecast_et0_3d PRIMA: max = {X_train.iloc[:, et0_idx].max():.2f} mm")
            print(f"   forecast_et0_3d DOPO:  max = {X_train_scaled[:, et0_idx].max():.2f} (normalizzato)")

        print(f"   ✅ Tutte le features ora hanno scala simile (media≈0, std≈1)")
        print(f"{'='*60}\n")

        # Initialize model
        # self.model = RandomForestRegressor(
        #     n_estimators=100,
        #     max_depth=15,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     random_state=42,
        #     n_jobs=-1
        # )

        # R2 basso
        # self.model = RandomForestRegressor(
        #     n_estimators=200,       
        #     max_depth=12,           
        #     max_features='sqrt',    
        #     min_samples_split=10,   
        #     min_samples_leaf=5,     
        #     random_state=42
        # )

        self.model = RandomForestRegressor(
            n_estimators=200,       # OK - più alberi = meglio
            max_depth=15,           # ✅ Aumentato (era 12)
            max_features=0.7,       # ✅ 70% features invece di sqrt (~21%)
            min_samples_split=5,    # ✅ Ridotto (era 10)
            min_samples_leaf=2,     # ✅ Ridotto (era 5)
            random_state=42,
            n_jobs=-1
        )

        # Train su dati SCALATI
        self.model.fit(X_train_scaled, y_train)

        print(f"🔍 Training shape: {X_train_scaled.shape}")
        print(f"🔍 Feature names: {len(self.feature_names)}")

        # Evaluate su dati SCALATI
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.training_metrics = {
            'r2_score': test_r2,
            'r2_train': train_r2,
            'mae_m3': test_mae,
            'rmse_m3': test_rmse,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X)
        }

        # Verifica scaler
        print(f"\n✅ VERIFICA SCALING:")
        print(f"   Scaler fitted: {hasattr(self.scaler, 'mean_')}")
        if hasattr(self.scaler, 'mean_'):
            print(f"   Scaler mean shape: {self.scaler.mean_.shape}")
            print(f"   Scaler scale shape: {self.scaler.scale_.shape}")
            print(f"   ✅ Scaler correttamente fitted su {len(self.feature_names)} features")
        else:
            print(f"   ❌ ERRORE: Scaler NON fitted!")

        return self.training_metrics

    def predict(self, features: pd.Series, forecast_data: Dict,
                historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Make prediction with ML model

        Args:
            features: Feature values
            forecast_data: Weather forecast
            historical_data: Optional historical data for comparison

        Returns:
            Dictionary with prediction and detailed explanation
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure features are in correct order
        X_pred = features[self.feature_names].values.reshape(1, -1)

        # ========== SCALA FEATURES PRIMA DELLA PREDIZIONE ==========
        # CRITICO: Deve usare STESSO scaler del training
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted! Model was not properly trained.")

        X_pred_scaled = self.scaler.transform(X_pred)

        # Predict su dati SCALATI
        water_m3 = self.model.predict(X_pred_scaled)[0]
        water_m3 = max(0, water_m3)  # No negative irrigation

        # Convert to mm
        water_mm = water_m3 / (self.FIELD_AREA_HA * 100)

        print("\n🔍 TEST SENSIBILITÀ FORECAST:")

        # Test 1: Raddoppia ET0
        X_test = X_pred.copy()
        X_test[0, 18] = X_pred[0, 18] * 2  # forecast_et0_3d × 2
        water_test = self.model.predict(X_test)[0]
        print(f"ET0 normale ({X_pred[0, 18]:.1f}mm): {water_m3:.1f} m³")
        print(f"ET0 doppio ({X_test[0, 18]:.1f}mm):  {water_test:.1f} m³")
        delta = abs(water_test - water_m3)
        print(f"Differenza: {delta:.1f} m³")

        if delta < 10:
            print("❌ PROBLEMA: Modello NON reagisce a cambio ET0!")
        else:
            print("✅ OK: Modello sensibile a ET0")

        # Test 2: Aggiungi pioggia
        X_test2 = X_pred.copy()
        X_test2[0, 17] = 50.0  # forecast_precipitation_3d = 50mm
        water_test2 = self.model.predict(X_test2)[0]
        print(f"\nPioggia 0mm: {water_m3:.1f} m³")
        print(f"Pioggia 50mm: {water_test2:.1f} m³")

        if abs(water_test2 - water_m3) < 10:
            print("❌ PROBLEMA: Modello NON reagisce a pioggia!")
        else:
            print("✅ OK: Modello sensibile a pioggia")

        # Calculate confidence and priority
        confidence = self._calculate_confidence(features)
        priority = self._calculate_priority(features, water_mm)

        # Generate comprehensive explanation
        explanation = self._generate_ml_explanation(
            features=features,
            water_mm=water_mm,
            forecast_data=forecast_data,
            historical_data=historical_data
        )

        # Current situation summary
        current_situation = self._summarize_current_situation(
            features, historical_data
        )

        # Forecast summary
        forecast_summary = self._summarize_forecast(forecast_data)

        # Historical comparison
        historical_comparison = None
        if historical_data is not None:
            historical_comparison = self._compare_with_history(
                water_mm, historical_data
            )

        # Feature importance for this prediction
        top_features = self._get_top_feature_importance(features, n=8)

        result = {
            'water_m3': water_m3,
            'water_mm': water_mm,
            'water_liters_m2': water_mm,
            'confidence': confidence,
            'priority': priority,
            'recommendation': self._generate_recommendation(water_mm, confidence, features),
            'current_situation': current_situation,
            'forecast_3days': forecast_summary,
            'explanation': explanation,
            'historical_comparison': historical_comparison,
            'feature_importance': top_features,
            'model_used': 'RandomForest'
        }

        return result

    def _calculate_confidence(self, features: pd.Series) -> str:
        """Calculate confidence level based on input data quality"""
        soil_moisture = features.get('avg_soil_moisture', 30)
        temp_min = features.get('forecast_temp_min', 10)

        if temp_min < 2:
            return "BASSA"
        elif soil_moisture < 25 or soil_moisture > 50:
            return "MEDIA"
        elif features.get('precipitation_sum_7d', 0) > 30:
            return "MEDIA"
        else:
            return "ALTA"

    def _calculate_priority(self, features: pd.Series, water_mm: float) -> str:
        """Calculate irrigation priority"""
        soil_moisture = features.get('avg_soil_moisture', 30)
        water_deficit = features.get('water_deficit', 0)

        if soil_moisture < 25:
            return "ALTA"
        elif soil_moisture < 30 and water_deficit > 5:
            return "ALTA"
        elif water_mm > 8:
            return "MEDIA"
        elif soil_moisture < 35:
            return "MEDIA"
        else:
            return "BASSA"

    def _generate_recommendation(self, water_mm: float, confidence: str,
                                features: pd.Series) -> str:
        """Generate human-readable recommendation"""
        temp_min = features.get('forecast_temp_min', 10)
        soil_moisture = features.get('avg_soil_moisture', 30)

        if water_mm < 2:
            rec = f"Nessuna irrigazione necessaria oggi ({water_mm:.1f}mm)"
        else:
            rec = f"Irrigare oggi con {water_mm:.1f}mm"

        if temp_min < 2:
            rec += " - ATTENZIONE: rischio gelo, irrigare nelle ore calde"
        elif soil_moisture < 25:
            rec += " - URGENTE: piante in stress idrico"

        return rec

    def _summarize_current_situation(self, features: pd.Series,
                                    historical_data: Optional[pd.DataFrame]) -> Dict:
        """Summarize current soil and irrigation situation"""
        soil_moisture = features.get('avg_soil_moisture', 0)
        soil_trend = features.get('soil_moisture_trend', 0)
        days_since = features.get('days_since_last_irrigation', 0)
        precip_7d = features.get('precipitation_sum_7d', 0)

        # Calculate days without rain
        days_without_rain = 0
        if precip_7d < 1:
            days_without_rain = 7
        elif features.get('precipitation_sum_3d', 0) < 0.5:
            days_without_rain = 3

        # Last irrigation amount
        last_irrigation_m3 = 0
        if historical_data is not None and len(historical_data) > 1:
            # Calculate from tfm1 difference
            tfm1_vals = historical_data['tfm1'].fillna(0).values
            if len(tfm1_vals) >= 2:
                last_irrigation_m3 = max(0, tfm1_vals[-1] - tfm1_vals[-2])

        return {
            'soil_moisture_avg': round(soil_moisture, 1),
            'soil_moisture_target': 40.0,
            'moisture_deficit': round(40.0 - soil_moisture, 1),
            'soil_trend_daily': round(soil_trend, 2),
            'days_without_rain': days_without_rain,
            'days_since_irrigation': int(days_since),
            'last_irrigation_m3': round(last_irrigation_m3, 0)
        }

    def _summarize_forecast(self, forecast_data: Dict) -> Dict:
        """Summarize weather forecast"""
        if forecast_data is None or 'daily' not in forecast_data:
            return {}

        daily = forecast_data['daily']
        n_days = min(3, len(daily['time']))

        rain_sum = sum(daily['precipitation_sum'][:n_days])
        et0_sum = sum(daily['et0_fao_evapotranspiration'][:n_days])
        temp_min = min(daily['temperature_2m_min'][:n_days])

        vpd_max = 0
        if 'vapor_pressure_deficit_max' in daily:
            vpd_max = max(daily['vapor_pressure_deficit_max'][:n_days])

        return {
            'rain_expected_mm': round(rain_sum, 2),
            'et0_expected_mm': round(et0_sum, 2),
            'vpd_max_kpa': round(vpd_max, 2),
            'temp_min_c': round(temp_min, 1),
            'frost_risk': temp_min < 2,
            'water_deficit_mm': round(et0_sum - rain_sum, 2)
        }

    def _compare_with_history(self, predicted_mm: float,
                             historical_data: pd.DataFrame) -> Dict:
        """Compare prediction with historical patterns"""
        if 'tfm1' not in historical_data.columns:
            return None

        # Calculate daily irrigation from tfm1
        tfm1_vals = historical_data['tfm1'].fillna(0).values
        daily_irrigation = np.diff(tfm1_vals)
        daily_irrigation = np.maximum(daily_irrigation, 0)

        # Convert to mm
        daily_irrigation_mm = daily_irrigation / (self.FIELD_AREA_HA * 100)

        # Statistics
        if len(daily_irrigation_mm) > 7:
            avg_7d = np.mean(daily_irrigation_mm[-7:])
        else:
            avg_7d = np.mean(daily_irrigation_mm) if len(daily_irrigation_mm) > 0 else 0

        if len(daily_irrigation_mm) > 14:
            avg_14d = np.mean(daily_irrigation_mm[-14:])
        else:
            avg_14d = avg_7d

        # Calculate differences
        diff_7d = ((predicted_mm - avg_7d) / avg_7d * 100) if avg_7d > 0 else 0
        diff_14d = ((predicted_mm - avg_14d) / avg_14d * 100) if avg_14d > 0 else 0

        # Trend explanation
        if abs(diff_7d) < 10:
            trend = "Coerente con media storica"
        elif diff_7d > 0:
            trend = f"Aumento del {abs(diff_7d):.0f}% rispetto alla media"
        else:
            trend = f"Riduzione del {abs(diff_7d):.0f}% rispetto alla media"

        return {
            'avg_irrigation_7d': round(avg_7d, 2),
            'avg_irrigation_14d': round(avg_14d, 2),
            'predicted_vs_avg_7d': round(diff_7d, 1),
            'predicted_vs_avg_14d': round(diff_14d, 1),
            'trend': trend
        }

    def _generate_ml_explanation(self, features: pd.Series, water_mm: float,
                                forecast_data: Dict,
                                historical_data: Optional[pd.DataFrame]) -> Dict:
        """Generate detailed ML explanation"""
        # Get historical averages
        hist_avg_7d = 0
        if historical_data is not None and 'tfm1' in historical_data.columns:
            tfm1_vals = historical_data['tfm1'].fillna(0).values
            daily_irr = np.diff(tfm1_vals)
            daily_irr = np.maximum(daily_irr, 0)
            daily_irr_mm = daily_irr / (self.FIELD_AREA_HA * 100)
            if len(daily_irr_mm) > 7:
                hist_avg_7d = np.mean(daily_irr_mm[-7:])

        # Summary
        soil_moisture = features.get('avg_soil_moisture', 30)
        et0 = features.get('forecast_et0_3d', 0)
        rain = features.get('forecast_precipitation_3d', 0)
        temp_min = features.get('forecast_temp_min', 10)

        summary_parts = []
        summary_parts.append(f"Raccomandazione ML: {water_mm:.1f}mm")

        if hist_avg_7d > 0:
            diff_pct = ((water_mm - hist_avg_7d) / hist_avg_7d * 100)
            if abs(diff_pct) > 10:
                summary_parts.append(
                    f"({'Aumento' if diff_pct > 0 else 'Riduzione'} "
                    f"del {abs(diff_pct):.0f}% rispetto alla media storica)"
                )

        if temp_min < 2:
            summary_parts.append(
                "Attenzione: temperatura minima prevista sotto 2°C - "
                "valutare irrigazione pomeridiana"
            )

        summary = ". ".join(summary_parts) + "."

        # Decision steps
        steps = [
            f"Analisi storica: media irrigazione {hist_avg_7d:.1f} mm/giorno ultimi 7 giorni",
            f"Deficit idrico previsto: {et0:.2f}mm (ET0) - {rain:.2f}mm (pioggia) = {et0 - rain:.2f}mm",
            f"Correzione umidità suolo: {soil_moisture:.1f}% vs 40% target",
            f"Modello ML analizza 17+ features storiche e previsionali",
            f"Fattori climatici: VPD, radiazione solare, trend umidità",
            f"TOTALE RACCOMANDATO: {water_mm:.1f}mm"
        ]

        # Key factors
        key_factors = self._get_top_feature_importance(features, n=4)

        # Warnings
        warnings = []
        if temp_min < 2:
            warnings.append("⚠️ ATTENZIONE: Temperatura minima prevista sotto 2°C")
            warnings.append("💡 Irrigare preferibilmente tra le 12:00 e le 16:00")
        if soil_moisture < 25:
            warnings.append("🚨 URGENTE: Piante in stress idrico!")
        if soil_moisture > 45:
            warnings.append("ℹ️ Umidità suolo già elevata")

        warnings.append("📊 Monitorare umidità suolo nelle prossime 24h")

        return {
            'summary': summary,
            'steps': steps,
            'key_factors': key_factors,
            'warnings': warnings
        }

    def _get_top_feature_importance(self, features: pd.Series, n: int = 5) -> list:
        """Get top N important features with their values"""
        if self.feature_importance is None:
            return []

        from .feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        descriptions = fe.get_feature_descriptions()

        top_features = []
        for _, row in self.feature_importance.head(n).iterrows():
            feat_name = row['feature']
            importance = row['importance']
            value = features.get(feat_name, 0)

            # Format value
            if feat_name in ['month', 'day_of_year', 'is_summer', 'days_since_last_irrigation']:
                value_str = f"{int(value)}"
            else:
                value_str = f"{value:.1f}"

            # Determine impact
            if importance > 0.15:
                impact = "Alto"
            elif importance > 0.08:
                impact = "Medio"
            else:
                impact = "Basso"

            top_features.append({
                'feature': feat_name,
                'importance': round(importance, 3),
                'value': value_str,
                'impact': impact,
                'weight': f"{importance * 100:.0f}%",
                'description': descriptions.get(feat_name, feat_name),
                'note': ""
            })

        return top_features

    def is_model_reliable(self) -> bool:
        """Check if model is reliable enough to use"""
        if self.training_metrics is None:
            return False
        return self.training_metrics['r2_score'] >= self.MIN_R2_SCORE

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
