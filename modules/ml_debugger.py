"""
ML Model Debugger
Diagnostica problemi nel modello e nelle predizioni
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLDebugger:
    """
    Debug completo del modello ML
    """

    def __init__(self):
        self.debug_logs = []
        self.warnings = []
        self.errors = []

    def debug_prediction(
        self,
        features: Dict[str, float],
        forecast_data: Dict[str, Any],
        model: Optional[Any],
        model_type: str,
        prediction: float,
        r2_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Debug completo di una predizione

        Args:
            features: Dictionary con tutte le features usate
            forecast_data: Dati forecast meteo
            model: Modello ML (se RandomForest)
            model_type: Tipo modello ('RandomForest' o 'Rule-Based')
            prediction: Valore predetto (mm)
            r2_score: R² score del modello

        Returns:
            report con analisi dettagliata
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': model_type,
            'prediction': float(prediction),
            'r2_score': float(r2_score) if r2_score else None,
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # 1. Verifica features
        feature_check = self._check_features(features, forecast_data)
        report['checks']['features'] = feature_check

        # 2. Verifica forecast impact
        forecast_check = self._check_forecast_impact(features, forecast_data, prediction)
        report['checks']['forecast'] = forecast_check

        # 3. Verifica sanity
        sanity_check = self._check_sanity(prediction, forecast_data, features)
        report['checks']['sanity'] = sanity_check

        # 4. Verifica modello
        if model is not None:
            model_check = self._check_model(model, features, prediction)
            report['checks']['model'] = model_check

        # 5. Genera warnings
        self._generate_warnings(report)

        # 6. Genera recommendations
        self._generate_recommendations(report, features, forecast_data, prediction)

        return report

    def _check_features(self, features: Dict, forecast_data: Dict) -> Dict:
        """
        Verifica che tutte le features necessarie siano presenti
        """
        result = {
            'total_features': len(features),
            'missing_critical': [],
            'present_features': list(features.keys()),
            'forecast_features_included': False
        }

        # FEATURES CRITICHE (devono esserci!)
        critical_features = [
            'forecast_et0_3d',
            'forecast_precipitation_3d',
            'avg_soil_moisture'
        ]

        # Verifica presenza
        for feat in critical_features:
            if feat not in features:
                result['missing_critical'].append(feat)
                logger.error(f"❌ Feature critica mancante: {feat}")

        # Verifica forecast incluso
        forecast_features = [k for k in features.keys() if 'forecast' in k]
        result['forecast_features_included'] = len(forecast_features) > 0
        result['forecast_features_count'] = len(forecast_features)
        result['forecast_features_list'] = forecast_features

        if not result['forecast_features_included']:
            logger.error("❌ CRITICO: Nessuna feature forecast inclusa!")
            result['error'] = "Previsioni meteo non incluse nelle features"

        # Verifica valori
        for key, value in features.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                result['missing_critical'].append(f"{key} (valore None/NaN)")

        return result

    def _check_forecast_impact(
        self,
        features: Dict,
        forecast_data: Dict,
        prediction: float
    ) -> Dict:
        """
        Verifica se le previsioni influenzano la predizione
        """
        result = {
            'forecast_data': forecast_data,
            'should_irrigate': None,
            'expected_range': None,
            'actual_prediction': float(prediction),
            'makes_sense': False,
            'reason': ''
        }

        # Estrai dati forecast dalla daily structure
        daily = forecast_data.get('daily', {})

        # Calcola somme per 3 giorni
        et0_values = daily.get('et0_fao_evapotranspiration', [])
        rain_values = daily.get('precipitation_sum', [])

        et0_sum = sum(et0_values[:3]) if et0_values else 0
        rain_sum = sum(rain_values[:3]) if rain_values else 0

        water_deficit = et0_sum - rain_sum

        # LOGICA DI VERIFICA

        # Caso 1: PIOGGIA ABBONDANTE (>30mm)
        if rain_sum > 30:
            result['should_irrigate'] = False
            result['expected_range'] = (0, 2)
            result['makes_sense'] = (0 <= prediction <= 2)
            result['reason'] = f"Pioggia abbondante ({rain_sum:.1f}mm) - non serve irrigare"

        # Caso 2: ET0 ALTISSIMO (>10mm) e NIENTE PIOGGIA
        elif et0_sum > 10 and rain_sum < 2:
            result['should_irrigate'] = True
            result['expected_range'] = (12, 20)
            result['makes_sense'] = (12 <= prediction <= 20)
            result['reason'] = f"ET0 molto alto ({et0_sum:.1f}mm), serve molta acqua"

        # Caso 3: DEFICIT ALTO (>8mm)
        elif water_deficit > 8:
            result['should_irrigate'] = True
            result['expected_range'] = (8, 15)
            result['makes_sense'] = (8 <= prediction <= 15)
            result['reason'] = f"Deficit idrico alto ({water_deficit:.1f}mm)"

        # Caso 4: SURPLUS (pioggia > ET0)
        elif water_deficit < -10:
            result['should_irrigate'] = False
            result['expected_range'] = (0, 1)
            result['makes_sense'] = (0 <= prediction <= 1)
            result['reason'] = f"Surplus idrico ({water_deficit:.1f}mm) - non serve irrigare"

        # Caso 5: NORMALE
        else:
            result['should_irrigate'] = True
            result['expected_range'] = (3, 8)
            result['makes_sense'] = (3 <= prediction <= 8)
            result['reason'] = "Condizioni normali"

        # Aggiungi dettagli
        result['water_deficit'] = float(water_deficit)
        result['et0_sum'] = float(et0_sum)
        result['rain_sum'] = float(rain_sum)

        # Flag se predizione non ha senso
        if not result['makes_sense']:
            logger.warning(
                f"⚠️ PREDIZIONE SOSPETTA: {prediction:.1f}mm "
                f"(atteso {result['expected_range'][0]}-{result['expected_range'][1]}mm). "
                f"Motivo: {result['reason']}"
            )

        return result

    def _check_sanity(
        self,
        prediction: float,
        forecast_data: Dict,
        features: Dict
    ) -> Dict:
        """
        Sanity checks sulla predizione
        """
        result = {
            'is_sane': True,
            'issues': []
        }

        # Check 1: Predizione negativa
        if prediction < 0:
            result['is_sane'] = False
            result['issues'].append({
                'type': 'negative_prediction',
                'severity': 'critical',
                'message': f"Predizione negativa: {prediction:.2f}mm"
            })

        # Check 2: Predizione > 25mm (molto raro)
        if prediction > 25:
            result['is_sane'] = False
            result['issues'].append({
                'type': 'extremely_high',
                'severity': 'warning',
                'message': f"Predizione molto alta: {prediction:.2f}mm (>25mm è raro)"
            })

        # Check 3: Predizione costante (sempre stesso valore)
        avg_irrigation = features.get('irrigation_avg_14d', 0)
        if avg_irrigation > 0 and abs(prediction - avg_irrigation) < 0.01:
            result['issues'].append({
                'type': 'constant_prediction',
                'severity': 'critical',
                'message': f"Predizione identica alla media storica ({avg_irrigation:.2f}mm). "
                          f"Il modello potrebbe non usare le features!"
            })

        # Check 4: Ignora previsioni meteo
        daily = forecast_data.get('daily', {})
        rain_values = daily.get('precipitation_sum', [])
        et0_values = daily.get('et0_fao_evapotranspiration', [])

        rain_sum = sum(rain_values[:3]) if rain_values else 0
        et0_sum = sum(et0_values[:3]) if et0_values else 0

        if rain_sum > 30 and prediction > 2:
            result['is_sane'] = False
            result['issues'].append({
                'type': 'ignores_heavy_rain',
                'severity': 'critical',
                'message': f"Pioggia prevista {rain_sum:.1f}mm ma raccomanda {prediction:.1f}mm. "
                          f"Il modello NON sta considerando la pioggia!"
            })

        if et0_sum > 10 and rain_sum < 2 and prediction < 8:
            result['is_sane'] = False
            result['issues'].append({
                'type': 'ignores_high_et0',
                'severity': 'critical',
                'message': f"ET0 alto ({et0_sum:.1f}mm), no pioggia, ma raccomanda solo {prediction:.1f}mm. "
                          f"Il modello NON sta considerando ET0!"
            })

        return result

    def _check_model(self, model, features: Dict, prediction: float) -> Dict:
        """
        Verifica stato del modello
        """
        result = {
            'model_class': model.__class__.__name__,
            'has_feature_importance': False,
            'top_features': [],
            'prediction_variance': None
        }

        # Check feature importance (se RandomForest)
        if hasattr(model, 'feature_importances_'):
            result['has_feature_importance'] = True

            importances = model.feature_importances_
            feature_names = list(features.keys())

            # Top 5 features
            if len(importances) == len(feature_names):
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                result['top_features'] = [(name, float(imp)) for name, imp in feature_importance[:5]]

                # Verifica che forecast features abbiano peso
                forecast_features_importance = [
                    (name, imp) for name, imp in feature_importance
                    if 'forecast' in name
                ]

                if not forecast_features_importance:
                    logger.error("❌ Nessuna feature forecast ha importance!")
                    result['error'] = "Forecast features hanno importance zero"
                elif sum(imp for _, imp in forecast_features_importance) < 0.1:
                    logger.warning(
                        f"⚠️ Feature forecast hanno bassa importance totale: "
                        f"{sum(imp for _, imp in forecast_features_importance):.3f}"
                    )
                    result['warning'] = f"Forecast features importance bassa: {sum(imp for _, imp in forecast_features_importance):.3f}"

        # Check se il modello è "stuck" (sempre stesso output)
        if hasattr(model, 'predict'):
            try:
                # Crea versione features con ET0 raddoppiato
                features_array = np.array([list(features.values())])
                pred_normal = model.predict(features_array)[0]

                # Perturba forecast ET0 (raddoppia)
                features_modified = features.copy()
                if 'forecast_et0_3d' in features_modified:
                    features_modified['forecast_et0_3d'] *= 2
                    features_array_mod = np.array([list(features_modified.values())])
                    pred_modified = model.predict(features_array_mod)[0]

                    variance = abs(pred_modified - pred_normal)
                    result['prediction_variance'] = float(variance)

                    if variance < 0.1:
                        logger.error(
                            f"❌ Modello NON reagisce a cambio ET0: "
                            f"{pred_normal:.2f} → {pred_modified:.2f}mm (Δ={variance:.3f})"
                        )
                        result['error'] = "Modello non sensibile a cambio forecast"
            except Exception as e:
                result['error'] = f"Errore test varianza: {str(e)}"

        return result

    def _generate_warnings(self, report: Dict):
        """
        Genera warnings basati sui check
        """
        warnings = []

        # Feature warnings
        if report['checks']['features']['missing_critical']:
            warnings.append({
                'level': 'critical',
                'message': f"Features critiche mancanti: {', '.join(report['checks']['features']['missing_critical'])}"
            })

        if not report['checks']['features']['forecast_features_included']:
            warnings.append({
                'level': 'critical',
                'message': "NESSUNA feature forecast inclusa! Il modello NON può considerare le previsioni meteo."
            })

        # Forecast impact warnings
        if not report['checks']['forecast']['makes_sense']:
            exp_range = report['checks']['forecast']['expected_range']
            warnings.append({
                'level': 'warning',
                'message': (
                    f"Predizione {report['prediction']:.1f}mm non coerente con forecast. "
                    f"Atteso: {exp_range[0]}-{exp_range[1]}mm. "
                    f"Motivo: {report['checks']['forecast']['reason']}"
                )
            })

        # Sanity warnings
        if not report['checks']['sanity']['is_sane']:
            for issue in report['checks']['sanity']['issues']:
                warnings.append({
                    'level': issue['severity'],
                    'message': issue['message']
                })

        report['warnings'] = warnings

    def _generate_recommendations(
        self,
        report: Dict,
        features: Dict,
        forecast_data: Dict,
        prediction: float
    ):
        """
        Genera raccomandazioni per fixare i problemi
        """
        recommendations = []

        # Rec 1: Features mancanti
        if report['checks']['features']['missing_critical']:
            recommendations.append({
                'priority': 'high',
                'action': 'add_missing_features',
                'details': (
                    f"Aggiungi features mancanti: {', '.join(report['checks']['features']['missing_critical'])}. "
                    f"Verifica che feature_engineering includa queste variabili."
                )
            })

        # Rec 2: Forecast non incluso
        if not report['checks']['features']['forecast_features_included']:
            recommendations.append({
                'priority': 'critical',
                'action': 'include_forecast_features',
                'details': (
                    "CRITICO: Aggiungi features forecast al dizionario prima di fare predict: "
                    "forecast_et0_3d, forecast_precipitation_3d, forecast_vpd_max, forecast_temp_min. "
                    "Verifica che add_forecast_features() venga chiamata correttamente."
                )
            })

        # Rec 3: Predizione non sensata
        if not report['checks']['forecast']['makes_sense']:
            # Calcola predizione corretta
            corrected = self._calculate_corrected_prediction(forecast_data, features)
            recommendations.append({
                'priority': 'high',
                'action': 'apply_correction',
                'details': (
                    f"Applica post-processing: predizione corretta dovrebbe essere ~{corrected:.1f}mm invece di {prediction:.1f}mm. "
                    f"Usa sanity checks o hybrid approach (ML + regole FAO-56)."
                )
            })

        # Rec 4: Modello non sensibile
        if 'model' in report['checks'] and report['checks']['model'].get('error'):
            recommendations.append({
                'priority': 'critical',
                'action': 'retrain_model',
                'details': (
                    f"Modello ha problemi: {report['checks']['model']['error']}. "
                    f"Considera retrain con più enfasi su features forecast o usa hybrid approach."
                )
            })

        report['recommendations'] = recommendations

    def _calculate_corrected_prediction(self, forecast_data: Dict, features: Dict) -> float:
        """
        Calcola predizione corretta usando regole fisiche
        """
        daily = forecast_data.get('daily', {})
        et0_values = daily.get('et0_fao_evapotranspiration', [])
        rain_values = daily.get('precipitation_sum', [])

        et0 = sum(et0_values[:3]) if et0_values else 0
        rain = sum(rain_values[:3]) if rain_values else 0
        moisture = features.get('avg_soil_moisture', 40)

        # Deficit base
        deficit = et0 - rain

        # Correzione umidità
        moisture_factor = (40 - moisture) / 40
        moisture_factor = max(0, min(2, moisture_factor))

        # Calcolo
        water = deficit * (0.7 + moisture_factor * 0.3)

        # Limiti
        water = max(0, min(20, water))

        return water

    def print_debug_report(self, report: Dict):
        """
        Stampa report leggibile su console
        """
        print("\n" + "=" * 80)
        print("🔍 ML DEBUG REPORT")
        print("=" * 80)

        print(f"\n📊 PREDIZIONE: {report['prediction']:.2f}mm")
        print(f"🤖 MODELLO: {report['model_type']}")
        if report['r2_score']:
            print(f"📈 R² SCORE: {report['r2_score']:.3f}")

        print("\n" + "-" * 80)
        print("✅ CHECKS")
        print("-" * 80)

        # Features
        feat_check = report['checks']['features']
        print(f"\n1. FEATURES ({feat_check['total_features']} totali)")
        print(f"   Forecast incluse: {'✅' if feat_check['forecast_features_included'] else '❌'}")
        if feat_check['forecast_features_list']:
            print(f"   Features forecast ({feat_check['forecast_features_count']}): {', '.join(feat_check['forecast_features_list'])}")
        if feat_check['missing_critical']:
            print(f"   ❌ MANCANTI: {feat_check['missing_critical']}")

        # Forecast impact
        forecast_check = report['checks']['forecast']
        print(f"\n2. FORECAST IMPACT")
        print(f"   ET0: {forecast_check['et0_sum']:.2f}mm")
        print(f"   Pioggia: {forecast_check['rain_sum']:.2f}mm")
        print(f"   Deficit: {forecast_check['water_deficit']:.2f}mm")
        print(f"   Range atteso: {forecast_check['expected_range']}mm")
        print(f"   Predizione sensata: {'✅' if forecast_check['makes_sense'] else '❌'}")
        print(f"   Motivo: {forecast_check['reason']}")

        # Sanity
        sanity_check = report['checks']['sanity']
        print(f"\n3. SANITY CHECK: {'✅' if sanity_check['is_sane'] else '❌'}")
        if sanity_check['issues']:
            for issue in sanity_check['issues']:
                emoji = "❌" if issue['severity'] == 'critical' else "⚠️"
                print(f"   {emoji} {issue['message']}")

        # Model
        if 'model' in report['checks']:
            model_check = report['checks']['model']
            print(f"\n4. MODEL CHECK")
            print(f"   Classe: {model_check['model_class']}")
            if model_check['top_features']:
                print(f"   Top 5 features:")
                for name, imp in model_check['top_features']:
                    print(f"      - {name}: {imp:.3f}")
            if model_check.get('prediction_variance') is not None:
                print(f"   Reattività: {model_check['prediction_variance']:.3f}mm")

        # Warnings
        if report['warnings']:
            print("\n" + "-" * 80)
            print("⚠️ WARNINGS")
            print("-" * 80)
            for i, warn in enumerate(report['warnings'], 1):
                emoji = "🔴" if warn['level'] == 'critical' else "🟡"
                print(f"{i}. {emoji} {warn['message']}")

        # Recommendations
        if report['recommendations']:
            print("\n" + "-" * 80)
            print("💡 RACCOMANDAZIONI")
            print("-" * 80)
            for i, rec in enumerate(report['recommendations'], 1):
                priority_emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(rec['priority'], '📌')
                print(f"{i}. {priority_emoji} [{rec['priority'].upper()}] {rec['action']}")
                print(f"   {rec['details']}")

        print("\n" + "=" * 80 + "\n")
