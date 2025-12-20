"""
API de Predicción - Store Sales Forecasting
=====================================================================

API Flask para realizar predicciones de ventas.
El pipeline espera TODAS las columnas del CSV original MENOS 'Sales'

Author: Grupo6
"""

import logging
import traceback
from datetime import datetime
from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd
import numpy as np
import joblib

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Cargar modelo
MODEL_PATH = "../models/stores_sales_forecasting_pipeline.pkl"
#try:
#    modelo_pipeline = joblib.load(MODEL_PATH)
#    logger.info(f"✅ Modelo cargado: {MODEL_PATH}")
#except Exception as e:
#    logger.error(f"❌ Error cargando modelo: {e}")
#    modelo_pipeline = None

try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    logger.info("✅ Modelo cargado: %s", MODEL_PATH)
except (FileNotFoundError, EOFError, ValueError) as e:
    logger.error("❌ Error cargando modelo: %s", str(e))
    modelo_pipeline = None

# COLUMNAS EXACTAS QUE ESPERA EL PIPELINE
# Son todas las del CSV original MENOS 'Sales' (21 columnas en total)
EXPECTED_COLUMNS = [
    'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
    'Customer ID', 'Customer Name', 'Segment', 'Country', 'City',
    'State', 'Branch', 'Postal Code', 'Region', 'Product ID',
    'Category', 'Sub-Category', 'Product Name',
    'Quantity', 'Discount', 'Profit'
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def prepare_dataframe_for_prediction(data, is_batch=False):
    """
    Prepara DataFrame con TODAS las 21 columnas que espera el pipeline.

    Args:
        data (dict or list): Datos de entrada del JSON
        is_batch (bool): Si es predicción batch

    Returns:
        pd.DataFrame: DataFrame con 21 columnas en el orden correcto
    """
    try:
        # Crear DataFrame inicial
        if is_batch:
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        logger.info(f"📊 DataFrame inicial: {df.shape}")
        logger.info(f"   Columnas recibidas: {list(df.columns)}")

        # Agregar columnas faltantes con valores por defecto
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                if col in ['Row ID', 'Postal Code']:
                    df[col] = 0
                elif col in ['Quantity', 'Discount', 'Profit']:
                    df[col] = 0.0
                elif col in ['Order Date', 'Ship Date']:
                    df[col] = '01/01/2020'  # Fecha por defecto
                else:
                    df[col] = 'Unknown'

        # Asegurar que Order Date y Ship Date sean strings
        if 'Order Date' in df.columns:
            df['Order Date'] = df['Order Date'].astype(str)
        if 'Ship Date' in df.columns:
            df['Ship Date'] = df['Ship Date'].astype(str)

        # Reordenar columnas en el orden exacto esperado
        df = df[EXPECTED_COLUMNS]

        logger.info(f"✅ DataFrame preparado: {df.shape}")
        logger.info(f"   Columnas finales: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"❌ Error preparando DataFrame: {e}")
        logger.error(traceback.format_exc())
        raise


def validate_input_data(data, is_batch=False):
    """Valida que los datos tengan los campos mínimos necesarios"""
    try:
        sample = data[0] if is_batch else data

        # Campos mínimos requeridos
        required_fields = ['Order Date', 'Ship Date', 'Quantity', 'Discount']
        missing = [f for f in required_fields if f not in sample]

        if missing:
            return False, f"Missing required fields: {missing}"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_model_metadata():
    """Obtiene metadata del modelo"""
    try:
        model = modelo_pipeline.steps[-1][1]
        model_name = modelo_pipeline.steps[-1][0]

        hyperparameters = {
            'model_type': type(model).__name__,
            'model_name': model_name
        }

        if hasattr(model, 'n_estimators'):
            hyperparameters['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            hyperparameters['max_depth'] = model.max_depth
        if hasattr(model, 'learning_rate'):
            hyperparameters['learning_rate'] = model.learning_rate
        if hasattr(model, 'random_state'):
            hyperparameters['random_state'] = model.random_state

        return {
            'hyperparameters': hyperparameters,
            'metrics': {
                'model_name': model_name,
                'note': 'Champion model from training'
            },
            'pipeline_info': {
                'total_steps': len(modelo_pipeline.steps),
                'steps': [s[0] for s in modelo_pipeline.steps]
            }
        }

    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return {'error': str(e)}


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Información de la API"""
    return jsonify({
        'api_name': 'Store Sales Forecasting API',
        'version': '3.0.0 - FINAL',
        'description': 'API de predicción de ventas',
        'endpoints': {
            '/': 'GET - Info',
            '/health': 'GET - Health check',
            '/predict_single': 'POST - Predicción individual',
            '/predict_batch': 'POST - Predicción batch'
        },
        'model_loaded': modelo_pipeline is not None,
        'expected_columns': len(EXPECTED_COLUMNS),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    model_ok = modelo_pipeline is not None
    return jsonify({
        'status': 'healthy' if model_ok else 'unhealthy',
        'model_loaded': model_ok,
        'timestamp': datetime.now().isoformat()
    }), 200 if model_ok else 503


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predicción individual"""
    request_time = datetime.now()

    try:
        if modelo_pipeline is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': request_time.isoformat()
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'timestamp': request_time.isoformat()
            }), 400

        logger.info(f"📥 Single prediction request")

        # Validar
        is_valid, msg = validate_input_data(data, is_batch=False)
        if not is_valid:
            return jsonify({
                'error': 'Invalid input',
                'message': msg,
                'timestamp': request_time.isoformat()
            }), 400

        # Preparar DataFrame con 21 columnas
        df = prepare_dataframe_for_prediction(data, is_batch=False)

        # Predecir
        pred_start = datetime.now()
        prediction = modelo_pipeline.predict(df)
        pred_time = (datetime.now() - pred_start).total_seconds()

        # Metadata
        metadata = get_model_metadata()

        # Respuesta
        response = {
            'status': 'success',
            'prediction': {
                'sales_forecast': float(prediction[0]),
                'currency': 'USD'
            },
            'input_summary': {
                'order_date': data.get('Order Date'),
                'category': data.get('Category'),
                'sub_category': data.get('Sub-Category'),
                'quantity': data.get('Quantity'),
                'discount': data.get('Discount'),
                'region': data.get('Region')
            },
            'model_info': metadata,
            'request_info': {
                'timestamp': request_time.isoformat(),
                'prediction_time_seconds': pred_time,
                'endpoint': 'predict_single'
            }
        }

        logger.info(f"✅ Prediction: ${prediction[0]:.2f}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Error processing request',
            'timestamp': request_time.isoformat(),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predicción batch"""
    request_time = datetime.now()

    try:
        if modelo_pipeline is None:
            return jsonify({
                'error': 'Model not loaded',
                'timestamp': request_time.isoformat()
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'timestamp': request_time.isoformat()
            }), 400

        logger.info("📥 Batch prediction: %d records", len(data))

        # Validar
        is_valid, msg = validate_input_data(data, is_batch=True)
        if not is_valid:
            return jsonify({
                'error': 'Invalid input',
                'message': msg,
                'timestamp': request_time.isoformat()
            }), 400

        # Preparar DataFrame
        df = prepare_dataframe_for_prediction(data, is_batch=True)

        # Tiempo de predicciones
        pred_start = datetime.now()
        predictions = modelo_pipeline.predict(df)
        pred_time = (datetime.now() - pred_start).total_seconds()

        # Metadata
        metadata = get_model_metadata()

        # Estadísticas
        stats = {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'median': float(np.median(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'std': float(np.std(predictions)),
            'total_forecast': float(np.sum(predictions))
        }

        # Lista de predicciones
        pred_list = []
        for idx, pred in enumerate(predictions):
            pred_list.append({
                'record_index': idx,
                'sales_forecast': float(pred),
                'input_summary': {
                    'category': data[idx].get('Category'),
                    'quantity': data[idx].get('Quantity'),
                    'discount': data[idx].get('Discount')
                }
            })

        # Respuesta
        response = {
            'status': 'success',
            'batch_info': {
                'total_records': len(predictions),
                'successful_predictions': len(predictions),
                'failed_predictions': 0
            },
            'predictions': pred_list,
            'predictions_statistics': stats,
            'model_info': metadata,
            'request_info': {
                'timestamp': request_time.isoformat(),
                'prediction_time_seconds': pred_time,
                'average_time_per_record': pred_time / len(predictions),
                'endpoint': 'predict_batch'
            }
        }

        logger.info(f"✅ Batch completed: {len(predictions)} predictions")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Error processing batch request',
            'timestamp': request_time.isoformat(),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 Store Sales Forecasting API v3.0 (FINAL)")
    logger.info("="*70)
    logger.info(f"   Model: {MODEL_PATH}")
    logger.info(f"   Loaded: {modelo_pipeline is not None}")
    logger.info(f"   Expected columns: {len(EXPECTED_COLUMNS)}")
    logger.info("="*70)

    app.run(debug=True, host='0.0.0.0', port=5000)