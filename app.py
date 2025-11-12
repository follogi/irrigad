"""
AI - Irrigation Optimization System
Flask Application
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import custom modules
from modules.data_loader import DataLoader
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.rule_based import RuleBasedPredictor
from modules.data_cleaning import DataCleaner
from modules.ml_debugger import MLDebugger


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'irrigad-ai-secret-key-2025'  # Change in production
CORS(app)

# Configuration
UPLOAD_FOLDER = 'venv/data/uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (increased for large CSV files)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage for data (in production use database or cache)
data_store = {
    'loader': None,
    'ml_model': None,
    'feature_engineer': None,
    'rule_based': None,
    'trained': False
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads and validation

    Expected files:
    - sm.csv: Soil moisture sensors
    - meteo.csv: Weather data
    - valve.csv: Irrigation data
    - previsioni.json: Weather forecast
    """
    try:
        # Check if all files are present
        required_files = ['sm', 'meteo', 'valve', 'previsioni']
        files_data = {}

        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({
                    'success': False,
                    'error': f'File {file_key} mancante. Carica tutti i 4 file richiesti.'
                }), 400

            file = request.files[file_key]

            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': f'Nessun file selezionato per {file_key}'
                }), 400

            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Estensione file non valida per {file_key}. Usa .csv o .json'
                }), 400

            # Save file
            filename = secure_filename(f"{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file.filename.rsplit('.', 1)[1].lower()}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            files_data[file_key] = filepath

        # Initialize DataLoader
        loader = DataLoader()
        all_errors = []

        # Load CSV files
        for file_type in ['sm', 'meteo', 'valve']:
            success, errors = loader.load_csv(files_data[file_type], file_type)
            if not success:
                all_errors.extend(errors)

        # Load JSON forecast
        success, errors = loader.load_json(files_data['previsioni'])
        if not success:
            all_errors.extend(errors)

        if all_errors:
            return jsonify({
                'success': False,
                'errors': all_errors
            }), 400

        # Validate periods overlap
        success, errors = loader.validate_periods()
        if not success:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400

        # DATA CLEANING - Pulisci outliers e anomalie
        cleaner = DataCleaner()
        loader.valve_df, loader.sm_df, loader.meteo_df, cleaning_report = cleaner.clean_all_data(
            valve_df=loader.valve_df,
            sm_df=loader.sm_df,
            meteo_df=loader.meteo_df
        )

        # Store loader in global storage (con dati puliti)
        data_store['loader'] = loader
        data_store['trained'] = False

        # Get summary
        summary = loader.get_summary()

        # Build response with cleaning report
        return jsonify({
            'success': True,
            'message': 'File caricati, validati e puliti con successo!',
            'summary': summary,
            'cleaning_report': {
                'total_outliers_removed': cleaning_report['summary']['total_outliers'],
                'valve': {
                    'outliers': cleaning_report['summary']['valve_outliers'],
                    'variance_reduction': f"{cleaning_report['details']['valve'].get('variance_reduction_pct', 0):.1f}%",
                    'negative_values': cleaning_report['details']['valve'].get('negative_values', 0),
                    'too_high_values': cleaning_report['details']['valve'].get('too_high_values', 0)
                },
                'sm': {
                    'outliers': cleaning_report['summary']['sm_outliers'],
                    'sensors_cleaned': cleaning_report['details']['sm'].get('sensors_checked', [])
                },
                'meteo': {
                    'outliers': cleaning_report['summary']['meteo_outliers'],
                    'sensors_cleaned': list(cleaning_report['details']['meteo'].get('outliers_by_sensor', {}).keys())
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore durante upload: {str(e)}'
        }), 500


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train ML model on uploaded data
    """
    try:
        # Check if data is loaded
        if data_store['loader'] is None:
            return jsonify({
                'success': False,
                'error': 'Nessun dato caricato. Carica i file prima di trainare.'
            }), 400

        loader = data_store['loader']

        # Get merged data
        merged_df = loader.get_merged_data()

        if len(merged_df) < 30:
            return jsonify({
                'success': False,
                'error': f'Dati insufficienti per training ({len(merged_df)} giorni). Minimo 30 giorni.'
            }), 400

        # Extract features
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.extract_features(merged_df)

        # Prepare training data
        X, y = feature_engineer.prepare_training_data()

        if len(X) < 30:
            return jsonify({
                'success': False,
                'error': f'Dati insufficienti dopo feature extraction ({len(X)} samples). Minimo 30.'
            }), 400

        # Train ML model
        ml_model = MLPredictor()
        training_metrics = ml_model.train(X, y)

        # Store models
        data_store['ml_model'] = ml_model
        data_store['feature_engineer'] = feature_engineer
        data_store['rule_based'] = RuleBasedPredictor()
        data_store['trained'] = True

        # Determine if ML model is reliable
        use_ml = ml_model.is_model_reliable()

        # Convert feature importance to native Python types
        feature_importance_list = ml_model.feature_importance.head(10).to_dict('records')
        feature_importance_list = convert_numpy_types(feature_importance_list)

        response_data = {
            'success': True,
            'message': 'Modello trainato con successo!',
            'training_metrics': convert_numpy_types(training_metrics),
            'use_ml': bool(use_ml),
            'model_type': 'RandomForest' if use_ml else 'Rule-Based',
            'feature_importance': feature_importance_list
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Errore durante training: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate irrigation recommendation
    """
    try:
        # Check if model is trained
        if not data_store['trained']:
            return jsonify({
                'success': False,
                'error': 'Modello non trainato. Esegui il training prima.'
            }), 400

        loader = data_store['loader']
        ml_model = data_store['ml_model']
        feature_engineer = data_store['feature_engineer']
        rule_based = data_store['rule_based']

        # Get merged data
        merged_df = loader.get_merged_data()
        forecast_data = loader.forecast_data

        # Prepare features for prediction
        features = feature_engineer.prepare_prediction_features(merged_df, forecast_data)

        # Decide which model to use
        use_ml = ml_model.is_model_reliable()

        if use_ml:
            # Use ML model
            result = ml_model.predict(features, forecast_data, merged_df)
            model_used = 'RandomForest'
            training_metrics = ml_model.training_metrics
        else:
            # Use rule-based model
            result = rule_based.predict(features, forecast_data)
            model_used = 'Rule-Based FAO-56'
            training_metrics = {
                'r2_score': 0.0,
                'mae_m3': 0.0,
                'rmse_m3': 0.0,
                'training_samples': len(merged_df),
                'note': 'Using rule-based fallback'
            }

        # Add metadata
        data_period = {
            'start': merged_df['day'].min().strftime('%Y-%m-%d'),
            'end': merged_df['day'].max().strftime('%Y-%m-%d'),
            'days': len(merged_df)
        }

        forecast_period = {
            'start': forecast_data['daily']['time'][0],
            'end': forecast_data['daily']['time'][-1]
        }

        # DEBUG PREDIZIONE - Analisi problemi ML
        debugger = MLDebugger()

        # Converti features Series in dict
        features_dict = features.to_dict() if hasattr(features, 'to_dict') else dict(features)

        # Esegui debug
        debug_report = debugger.debug_prediction(
            features=features_dict,
            forecast_data=forecast_data,
            model=ml_model.model if use_ml else None,
            model_type=model_used,
            prediction=result['water_mm'],
            r2_score=training_metrics.get('r2_score', None) if use_ml else None
        )

        # Stampa report su console (backend)
        debugger.print_debug_report(debug_report)

        # Build complete response
        response = {
            'success': True,
            'model_used': model_used,
            'training_metrics': training_metrics,
            'prediction': {
                'water_m3': round(result['water_m3'], 1),
                'water_mm': round(result['water_mm'], 2),
                'water_liters_m2': round(result['water_mm'], 2),
                'confidence': result['confidence'],
                'priority': result['priority'],
                'recommendation': result.get('recommendation', f"Irrigare con {result['water_mm']:.1f}mm")
            },
            'current_situation': result.get('current_situation', {}),
            'forecast_3days': result.get('forecast_3days', {}),
            'decision_explanation': result.get('explanation', {}),
            'historical_comparison': result.get('historical_comparison', {}),
            'feature_importance': result.get('feature_importance', []),
            'metadata': {
                'calculation_date': datetime.now().isoformat(),
                'data_period': data_period,
                'forecast_period': forecast_period,
                'algorithm_version': '1.0.0',
                'model_type': model_used
            },
            # Aggiungi debug info
            'debug': {
                'warnings': debug_report['warnings'],
                'recommendations': debug_report['recommendations'],
                'sanity_check': debug_report['checks']['sanity']['is_sane'],
                'forecast_included': debug_report['checks']['features']['forecast_features_included'],
                'checks': debug_report['checks']
            }
        }

        # Convert numpy types to native Python types
        response = convert_numpy_types(response)

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Errore durante predizione: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/analytics', methods=['GET'])
def analytics():
    """
    Return data for charts and analytics
    """
    try:
        if data_store['loader'] is None:
            return jsonify({
                'success': False,
                'error': 'Nessun dato caricato'
            }), 400

        loader = data_store['loader']
        merged_df = loader.get_merged_data()

        # Prepare time series data
        merged_df['date_str'] = pd.to_datetime(merged_df['day']).dt.strftime('%Y-%m-%d')

        # Soil moisture trend
        sm_trend = {
            'dates': merged_df['date_str'].tolist(),
            'sm1': merged_df['sm1'].fillna(0).tolist(),
            'sm2': merged_df['sm2'].fillna(0).tolist(),
            'avg': ((merged_df['sm1'].fillna(0) + merged_df['sm2'].fillna(0)) / 2).tolist()
        }

        # Irrigation history
        tfm1_vals = merged_df['tfm1'].fillna(0).values
        daily_irrigation = np.diff(tfm1_vals)
        daily_irrigation = np.maximum(daily_irrigation, 0)
        daily_irrigation_mm = daily_irrigation / 1000  # Convert to mm (approximation)

        irrigation_history = {
            'dates': merged_df['date_str'].tolist()[1:],  # Skip first day
            'values': daily_irrigation_mm.tolist()
        }

        # Weather data
        weather_data = {
            'dates': merged_df['date_str'].tolist(),
            'precipitation': merged_df['precipitation'].fillna(0).tolist(),
            'temperature': merged_df['airtemperaturemean'].fillna(0).tolist()
        }

        # Feature importance (if model trained)
        feature_importance_data = []
        if data_store['ml_model'] is not None:
            fi = data_store['ml_model'].feature_importance.head(10)
            feature_importance_data = fi.to_dict('records')

        response_data = {
            'success': True,
            'soil_moisture_trend': sm_trend,
            'irrigation_history': irrigation_history,
            'weather_data': weather_data,
            'feature_importance': feature_importance_data
        }

        # Convert numpy types to native Python types
        response_data = convert_numpy_types(response_data)

        return jsonify(response_data)

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Errore durante analisi: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': data_store['loader'] is not None,
        'model_trained': data_store['trained']
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File troppo grande. Dimensione massima consentita: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB. Riduci la dimensione del file o contatta il supporto.'
    }), 413


if __name__ == '__main__':
    print("=" * 60)
    print("AI - Irrigation Optimization System")
    print("=" * 60)
    print(f"Server starting on http://localhost:5000")
    print("Upload your data files to get started!")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
