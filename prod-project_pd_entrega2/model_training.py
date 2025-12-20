"""
Script: Model Training con MLflow
Descripción: Entrenamiento de múltiples modelos de regresión con tracking en MLflow
Autor: Grupo6
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import logging
import warnings
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ============================================================================
# 2. CONSTANTES
# ============================================================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "Sales_Forecasting_Model_Selection"
LOG_FILE = "ml_system.log"
DATA_PATH = '../data/raw/stores_sales_forecasting_updated_v3.1.csv'
PIPELINE_PATH = '../models/stores_sales_forecasting_data_pre_proc.pkl'
OUTPUT_PIPELINE_PATH = '../models/stores_sales_forecasting_pipeline.pkl'
RESULTS_PATH = '../results/models_comparison.csv'
CHAMPION_SUMMARY_PATH = 'champion_summary.json'
CV_FOLDS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 2026

MODEL_CONFIGURATIONS = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {},
        'description': 'Modelo de regresión lineal básico'
    },
    'random_forest': {
        'model': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'n_estimators': 200,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'max_features': 1.0,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'description': 'Random Forest con 200 árboles'
    },
    'gradient_boosting': {
        'model': GradientBoostingRegressor(random_state=2024, n_estimators=100, learning_rate=0.1),
        'params': {
            'random_state': 2024,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'description': 'Gradient Boosting con tasa de aprendizaje 0.1'
    },
    'svr': {
        'model': SVR(kernel='rbf', C=10, epsilon=0.1),
        'params': {
            'kernel': 'rbf',
            'C': 10,
            'epsilon': 0.1,
            'gamma': 'scale'
        },
        'description': 'Support Vector Regressor con kernel RBF'
    },
    'xgboost': {
        'model': XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'random_state': RANDOM_STATE,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        },
        'description': 'XGBoost optimizado para forecasting'
    }
}


# ============================================================================
# 3. CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    filename=LOG_FILE,
    encoding="utf-8",
    filemode="a",
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("Iniciando proceso de entrenamiento de modelos")
logger.info("=" * 80)


# ============================================================================
# 4. CONFIGURACIÓN DE MLFLOW
# ============================================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Obtener el experimento
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

print(f"✅ MLflow configurado")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experimento: {EXPERIMENT_NAME}")
print(f"   Experiment ID: {experiment_id}")


# ============================================================================
# 5. INFORMACIÓN DE MODELOS CONFIGURADOS
# ============================================================================
print(f"\n✅ Configurados {len(MODEL_CONFIGURATIONS)} modelos:")
for name, config in MODEL_CONFIGURATIONS.items():
    print(f"   - {name}: {config['description']}")


# ============================================================================
# 6. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
dataset = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')

print(f"\n✅ Dataset cargado: {dataset.shape}")

# Seleccionar features numéricas (excluyendo target)
numeric_columns = dataset.select_dtypes(
    include=['int64', 'float64', 'int32', 'float32']
).columns.tolist()

if 'Sales' in numeric_columns:
    numeric_columns.remove('Sales')

X = dataset[numeric_columns].copy()
y = dataset['Sales'].copy()

print(f"   Features: {X.shape[1]} columnas")
print(f"   Registros: {len(X):,}")
print(f"   Target (Sales): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")


# ============================================================================
# 7. ENTRENAMIENTO DE MODELOS CON MLFLOW TRACKING
# ============================================================================
results = {}
run_ids = {}

logger.info("Iniciando entrenamiento de modelos")
start_time = time.time()

print("\n" + "=" * 80)
print("ENTRENAMIENTO DE MODELOS CANDIDATOS (CHALLENGERS)")
print("=" * 80 + "\n")

current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
formatted_time

with mlflow.start_run(run_name=formatted_time):
    for model_name, config in MODEL_CONFIGURATIONS.items():
        print(f"\n🔄 Entrenando: {model_name}")
        print(f"   Descripción: {config['description']}")

        # Iniciar run de MLflow
        with mlflow.start_run(run_name=f"{model_name}_challenger", nested=True) as run:
            run_id = run.info.run_id
            run_ids[model_name] = run_id

            # Tiempo de inicio
            model_start = time.time()

            # ============================================
            # 1. REGISTRAR HIPERPARÁMETROS
            # ============================================
            print("   📋 Registrando hiperparámetros...")
            mlflow.log_params(config['params'])

            # Parámetros adicionales del experimento
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("random_state", config['params'].get('random_state', 'N/A'))
            mlflow.log_param("dataset_size", len(X))
            mlflow.log_param("n_features", X.shape[1])

            # ============================================
            # 2. ENTRENAR MODELO CON CROSS-VALIDATION
            # ============================================
            print("   🎯 Realizando validación cruzada...")
            cv_scores = cross_val_score(
                config['model'],
                X,
                y,
                scoring='neg_root_mean_squared_error',
                cv=CV_FOLDS,
                n_jobs=-1
            )

            # Convertir a valores positivos
            rmse_scores = -cv_scores
            rmse_mean = rmse_scores.mean()
            rmse_std = rmse_scores.std()

            # Calcular otras métricas
            cv_r2 = cross_val_score(
                config['model'],
                X,
                y,
                scoring='r2',
                cv=CV_FOLDS,
                n_jobs=-1
            ).mean()

            cv_mae = -cross_val_score(
                config['model'],
                X,
                y,
                scoring='neg_mean_absolute_error',
                cv=CV_FOLDS,
                n_jobs=-1
            ).mean()

            # Tiempo de entrenamiento
            training_time = time.time() - model_start

            # Guardar resultados
            results[model_name] = {
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'r2': cv_r2,
                'mae': cv_mae,
                'training_time': training_time,
                'run_id': run_id
            }

            # ============================================
            # 3. REGISTRAR MÉTRICAS
            # ============================================
            print("   📊 Registrando métricas...")
            mlflow.log_metric("rmse_mean", rmse_mean)
            mlflow.log_metric("rmse_std", rmse_std)
            mlflow.log_metric("r2_score", cv_r2)
            mlflow.log_metric("mae", cv_mae)
            mlflow.log_metric("training_time_seconds", training_time)

            # Registrar métricas individuales de CV
            for fold, score in enumerate(rmse_scores, 1):
                mlflow.log_metric(f"rmse_fold_{fold}", score)

            # ============================================
            # 4. REGISTRAR TAGS
            # ============================================
            mlflow.set_tags({
                "model_type": model_name,
                "model_status": "challenger",
                "description": config['description'],
                "framework": "sklearn" if model_name != 'xgboost' else "xgboost",
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cv_strategy": "10-fold"
            })

            # ============================================
            # 5. LOGGING Y PRINT
            # ============================================
            logger.info(
                f"{model_name} - RMSE: {rmse_mean:.2f} (+/- {rmse_std:.2f}), "
                f"R²: {cv_r2:.4f}, MAE: {cv_mae:.2f}, Tiempo: {training_time:.2f}s"
            )

            print(f"   ✅ Completado:")
            print(f"      RMSE: {rmse_mean:.2f} (+/- {rmse_std:.2f})")
            print(f"      R² Score: {cv_r2:.4f}")
            print(f"      MAE: {cv_mae:.2f}")
            print(f"      Tiempo: {training_time:.2f}s")
            print(f"      Run ID: {run_id[:8]}...")

    mlflow.end_run()

mlflow.end_run()

total_time = time.time() - start_time
logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")

print(f"\n✅ Todos los modelos entrenados en {total_time:.2f} segundos")


# ============================================================================
# 8. COMPARACIÓN DE RESULTADOS
# ============================================================================
# Crear DataFrame de resultados
df_results = pd.DataFrame([
    {
        'model': name,
        'rmse_mean': results[name]['rmse_mean'],
        'rmse_std': results[name]['rmse_std'],
        'r2_score': results[name]['r2'],
        'mae': results[name]['mae'],
        'training_time': results[name]['training_time'],
        'run_id': results[name]['run_id']
    }
    for name in results
]).sort_values('rmse_mean')

print("\n" + "=" * 80)
print("TABLA COMPARATIVA DE MODELOS (ordenados por RMSE)")
print("=" * 80 + "\n")
print(df_results.to_string(index=False))

# Identificar el modelo campeón
champion_name = df_results.iloc[0]['model']
champion_rmse = df_results.iloc[0]['rmse_mean']
champion_r2 = df_results.iloc[0]['r2_score']
champion_run_id = df_results.iloc[0]['run_id']

print(f"\n🏆 MODELO CAMPEÓN: {champion_name.upper()}")
print(f"   RMSE: {champion_rmse:.2f}")
print(f"   R² Score: {champion_r2:.4f}")
print(f"   Run ID: {champion_run_id}")

logger.info(f"Modelo campeón seleccionado: {champion_name} con RMSE={champion_rmse:.2f}")


# ============================================================================
# 9. REGISTRO DEL MODELO CAMPEÓN
# ============================================================================
print("\n" + "=" * 80)
print("REGISTRO DEL MODELO CAMPEÓN")
print("=" * 80 + "\n")

# Obtener configuración del modelo campeón
champion_config = MODEL_CONFIGURATIONS[champion_name]

# Crear nuevo run para el modelo campeón
with mlflow.start_run(run_name=f"{champion_name}_champion") as run:
    champion_final_run_id = run.info.run_id

    print(f"🏆 Entrenando modelo campeón: {champion_name}")

    # ============================================
    # 1. ENTRENAR EN TODO EL DATASET
    # ============================================
    champion_model = champion_config['model']
    champion_model.fit(X, y)

    # Predicciones y métricas en training set
    y_pred = champion_model.predict(X)
    final_rmse = np.sqrt(mean_squared_error(y, y_pred))
    final_r2 = r2_score(y, y_pred)
    final_mae = mean_absolute_error(y, y_pred)

    # ============================================
    # 2. REGISTRAR HIPERPARÁMETROS
    # ============================================
    mlflow.log_params(champion_config['params'])
    mlflow.log_param("dataset_size", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("training_type", "full_dataset")

    # ============================================
    # 3. REGISTRAR MÉTRICAS
    # ============================================
    mlflow.log_metric("rmse", final_rmse)
    mlflow.log_metric("r2_score", final_r2)
    mlflow.log_metric("mae", final_mae)
    mlflow.log_metric("cv_rmse_mean", champion_rmse)  # Métrica de CV
    mlflow.log_metric("cv_r2_score", champion_r2)     # Métrica de CV

    # ============================================
    # 4. REGISTRAR TAGS
    # ============================================
    mlflow.set_tags({
        "model_type": champion_name,
        "model_status": "champion",
        "description": champion_config['description'],
        "framework": "sklearn" if champion_name != 'xgboost' else "xgboost",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_criteria": "lowest_cv_rmse",
        "previous_run_id": champion_run_id
    })

    # ============================================
    # 5. REGISTRAR EL MODELO
    # ============================================
    print("   📦 Registrando modelo en MLflow...")

    # Inferir signature del modelo
    signature = infer_signature(X, y_pred)

    # Registrar modelo
    if champion_name == 'xgboost':
        mlflow.xgboost.log_model(
            champion_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="sales_forecasting_champion"
        )
    else:
        mlflow.sklearn.log_model(
            champion_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="sales_forecasting_champion"
        )

    # ============================================
    # 6. REGISTRAR ARTEFACTOS ADICIONALES
    # ============================================
    # Guardar resumen de resultados
    results_summary = {
        'champion_model': champion_name,
        'cv_rmse': champion_rmse,
        'cv_r2': champion_r2,
        'final_rmse': final_rmse,
        'final_r2': final_r2,
        'final_mae': final_mae,
        'training_date': datetime.now().isoformat(),
        'hyperparameters': champion_config['params']
    }

    with open(CHAMPION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    mlflow.log_artifact(CHAMPION_SUMMARY_PATH)

    # Guardar tabla de comparación
    df_results.to_csv(RESULTS_PATH, index=False)
    mlflow.log_artifact(RESULTS_PATH)

    print(f"\n✅ Modelo campeón registrado exitosamente")
    print(f"   Run ID: {champion_final_run_id}")
    print(f"   Registered Model: sales_forecasting_champion")
    print(f"   RMSE (training): {final_rmse:.2f}")
    print(f"   R² (training): {final_r2:.4f}")
    print(f"   MAE (training): {final_mae:.2f}")

    logger.info(f"Modelo campeón registrado - Run ID: {champion_final_run_id}")
    logger.info(f"Métricas finales - RMSE: {final_rmse:.2f}, R²: {final_r2:.4f}")


# ============================================================================
# 10. INTEGRACIÓN CON PIPELINE
# ============================================================================
print("\n" + "=" * 80)
print("INTEGRACIÓN CON PIPELINE DE PREPROCESAMIENTO")
print("=" * 80 + "\n")

# Cargar pipeline de preprocesamiento
pipeline = joblib.load(PIPELINE_PATH)
print("✅ Pipeline de preprocesamiento cargado")

# Agregar modelo campeón al pipeline
pipeline.steps.append((champion_name, champion_model))
print(f"✅ Modelo {champion_name} agregado al pipeline")

# Preparar datos para reentrenamiento
data_train = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')

x_full = data_train.drop(['Sales'], axis=1)
y_full = data_train['Sales']

# Split temporal (80/20) - SIN shuffle para mantener orden temporal
split_index = int(len(data_train) * (1 - TEST_SIZE))

x_train = x_full.iloc[:split_index].copy()
x_test = x_full.iloc[split_index:].copy()
y_train = y_full.iloc[:split_index].copy()
y_test = y_full.iloc[split_index:].copy()

print(f"\n📊 Datos divididos:")
print(f"   Train: {len(x_train):,} registros ({len(x_train)/len(x_full)*100:.1f}%)")
print(f"   Test: {len(x_test):,} registros ({len(x_test)/len(x_full)*100:.1f}%)")

# Entrenar pipeline completo
print("\n🔄 Entrenando pipeline completo...")
pipeline.fit(x_train, y_train)

# Evaluar en conjunto de test
y_pred_test = pipeline.predict(x_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n📊 Métricas en conjunto de test:")
print(f"   RMSE: {test_rmse:.2f}")
print(f"   R² Score: {test_r2:.4f}")
print(f"   MAE: {test_mae:.2f}")

# Guardar pipeline completo
joblib.dump(pipeline, OUTPUT_PIPELINE_PATH)
print(f"\n✅ Pipeline completo guardado en: {OUTPUT_PIPELINE_PATH}")

logger.info(f"Pipeline completo guardado con modelo campeón: {champion_name}")
logger.info(f"Métricas en test - RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")


# ============================================================================
# 11. RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 100)
print(" " * 35 + "RESUMEN FINAL")
print("=" * 100 + "\n")

print("📊 MODELOS EVALUADOS:")
for i, row in df_results.iterrows():
    status = "🏆 CHAMPION" if row['model'] == champion_name else "🔵 Challenger"
    print(f"   {status} {row['model']:20s} - RMSE: {row['rmse_mean']:7.2f} | R²: {row['r2_score']:.4f}")

print(f"\n🏆 MODELO SELECCIONADO: {champion_name.upper()}")
print(f"   Validación Cruzada RMSE: {champion_rmse:.2f}")
print(f"   Test Set RMSE: {test_rmse:.2f}")
print(f"   Test Set R²: {test_r2:.4f}")

print(f"\n📦 ARTEFACTOS GENERADOS:")
print(f"   ✓ Pipeline completo: {OUTPUT_PIPELINE_PATH}")
print(f"   ✓ Modelo registrado en MLflow: sales_forecasting_champion")
print(f"   ✓ Comparación de modelos: {RESULTS_PATH}")
print(f"   ✓ Resumen del campeón: {CHAMPION_SUMMARY_PATH}")

print(f"\n🔗 MLFLOW TRACKING:")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experimento: {EXPERIMENT_NAME}")
print(f"   Total de runs: {len(results) + 1}")
print(f"   Champion Run ID: {champion_final_run_id}")

print("\n" + "=" * 100)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 100 + "\n")

logger.info("=" * 80)
logger.info("Proceso de entrenamiento completado exitosamente")
logger.info(f"Modelo campeón: {champion_name}")
logger.info(f"RMSE en test: {test_rmse:.2f}")
logger.info("=" * 80)