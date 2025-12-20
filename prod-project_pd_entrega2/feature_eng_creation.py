"""
Script de Feature Engineering y Preprocesamiento de Datos
=========================================================

Este módulo maneja la creación de features derivadas y el pipeline completo
de preprocesamiento de datos para el proyecto Store Sales Forecasting.

Incluye:
- Creación de features temporales
- Imputación de valores faltantes
- Codificación de variables categóricas
- Transformaciones logarítmicas
- Normalización
- Tracking con MLflow

Author: Grupo6
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.transformation import LogTransformer
from feature_engine.selection import DropFeatures

import mlflow
import mlflow.sklearn

# Importar operadores personalizados
import operators as ops

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    filename="ml_system.log",
    encoding="utf-8",
    filemode="a",
    level=logging.INFO,
    format="{asctime} | {levelname:8s} | {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE VARIABLES GLOBALES
# ============================================================================

# Variables categóricas con valores faltantes - Imputación por 'Missing'
CATEGORICAL_VARS_WITH_NA_MISSING = ['Segment']

# Variables categóricas con valores faltantes - Imputación por frecuencia
CATEGORICAL_VARS_WITH_NA_FREQUENT = ['Sub-Category']

# Variables numéricas con valores faltantes - Imputación por media
NUMERICAL_VARS_WITH_NA = ['Quantity', 'Discount']

# Variables a eliminar (no aportan al modelo)
DROP_FEATURES = [
    'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
    'Order Date', 'Ship Date', 'Branch', 'Postal Code',
    'Product ID', 'Product Name'
]

# Variables para transformación logarítmica
NUMERICAL_LOG_VARS = ['Quantity']

# Variables para codificación ordinal (calidad/prioridad)
QUAL_VARS = ['Ship Mode']

# Mapeos para variables categóricas ordinales
QUAL_MAPPINGS = {
    'Standard Class': 1,
    'Second Class': 2,
    'First Class': 3
}

# Variables categóricas para frequency encoding
CATEGORICAL_VARS = [
    'Segment', 'Sub-Category', 'Country', 'City',
    'State', 'Region', 'Category'
]

# Features finales para el modelo (sin Profit para evitar data leakage)
FEATURES = [
    'Quantity', 'Discount', 'Ship Mode', 'Segment',
    'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category',
    'Order_Month', 'Order_Quarter', 'Days to Ship'
]

# Configuración de rutas
DATA_RAW_PATH = '../data/raw/stores_sales_forecasting_updated_v3.1.csv'
DATA_INTERIM_PATH = '../data/interim/'
MODELS_PATH = '../models/'
PIPELINE_FILENAME = 'stores_sales_forecasting_data_pre_proc.pkl'

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Feature_Engineering_Pipeline"

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def setup_mlflow():
    """
    Configura MLflow para tracking de experimentos.

    Returns:
        str: ID del experimento creado/encontrado
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow configurado - Experimento: {MLFLOW_EXPERIMENT_NAME}")
        return experiment.experiment_id
    except Exception as e:
        logger.error(f"Error configurando MLflow: {e}")
        raise


def create_directories():
    """
    Crea los directorios necesarios si no existen.
    """
    directories = [DATA_INTERIM_PATH, MODELS_PATH]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio verificado/creado: {directory}")


def print_section_header(title):
    """
    Imprime un encabezado formateado para las secciones.

    Args:
        title (str): Título de la sección
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"{title.center(70)}")
    print(f"{separator}\n")


# ============================================================================
# FUNCIONES DE CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

def load_and_prepare_data(filepath=DATA_RAW_PATH):
    """
    Carga y prepara el dataset inicial, creando features derivadas.

    Args:
        filepath (str): Ruta al archivo CSV de datos

    Returns:
        pd.DataFrame: Dataset con features derivadas
    """
    print_section_header("1. CARGA Y PREPARACIÓN DE DATOS")

    try:
        # Cargar dataset
        logger.info(f"Cargando dataset desde: {filepath}")
        data = pd.read_csv(filepath, sep=';', encoding='utf-8')
        print(f"✅ Dataset cargado: {data.shape}")
        logger.info(f"Dataset cargado exitosamente: {data.shape}")

        # Convertir fechas
        print("\n📅 Convirtiendo columnas de fechas...")
        data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True, errors='coerce')
        data['Ship Date'] = pd.to_datetime(data['Ship Date'], dayfirst=True, errors='coerce')

        # Crear features derivadas
        print("🔧 Creando features derivadas...")
        data['Order_Month'] = data['Order Date'].dt.month
        data['Order_Quarter'] = data['Order Date'].dt.quarter
        data['Days to Ship'] = (data['Ship Date'] - data['Order Date']).dt.days

        # Convertir a numérico
        print("🔢 Convirtiendo variables numéricas...")
        numeric_cols = [
            'Postal Code', 'Discount', 'Quantity',
            'Order_Month', 'Order_Quarter', 'Days to Ship'
        ]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"\n✅ Features derivadas creadas:")
        print(f"   - Order_Month: Mes de la orden")
        print(f"   - Order_Quarter: Trimestre de la orden")
        print(f"   - Days to Ship: Días entre orden y envío")
        print(f"\n📊 Dataset final: {data.shape}")

        logger.info("Features derivadas creadas exitosamente")
        return data

    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        raise


def create_artificial_nulls(data, null_percentage=0.05, random_seed=42):
    """
    Crea valores nulos artificiales para demostrar el pipeline.

    Args:
        data (pd.DataFrame): Dataset original
        null_percentage (float): Porcentaje de nulos a crear (default: 0.05)
        random_seed (int): Semilla para reproducibilidad

    Returns:
        pd.DataFrame: Dataset con nulos artificiales
    """
    print_section_header("2. CREACIÓN DE VALORES NULOS ARTIFICIALES")

    data_copy = data.copy()
    np.random.seed(random_seed)

    # Verificar si ya existen nulos
    existing_nulls = data_copy.isnull().sum()
    has_nulls = existing_nulls[existing_nulls > 0]

    if len(has_nulls) == 0:
        print("⚠️  No hay valores nulos. Creando algunos artificialmente...\n")

        sample_size = int(len(data_copy) * null_percentage)

        # Crear nulos en variables categóricas
        for var in ['Ship Mode', 'Segment', 'Sub-Category']:
            null_indices = np.random.choice(data_copy.index, sample_size, replace=False)
            data_copy.loc[null_indices, var] = np.nan
            print(f"   ✓ {var}: {sample_size} nulos creados")

        # Crear nulos en variables numéricas
        for var in ['Quantity', 'Discount']:
            null_indices = np.random.choice(data_copy.index, sample_size, replace=False)
            data_copy.loc[null_indices, var] = np.nan
            print(f"   ✓ {var}: {sample_size} nulos creados")

        logger.info(f"Valores nulos artificiales creados ({null_percentage*100}%)")
    else:
        print("ℹ️  El dataset ya contiene valores nulos\n")
        logger.info("Dataset ya contenía valores nulos")

    # Mostrar resumen de nulos
    print("\n📊 RESUMEN DE VALORES NULOS:")
    null_summary = data_copy.isnull().sum()
    null_summary = null_summary[null_summary > 0].sort_values(ascending=False)

    if len(null_summary) > 0:
        for col, count in null_summary.items():
            pct = (count / len(data_copy)) * 100
            print(f"   {col:20s}: {count:5d} ({pct:5.2f}%)")
    else:
        print("   No hay valores nulos en el dataset")

    return data_copy


# ============================================================================
# FUNCIONES DE SPLIT Y CONFIGURACIÓN
# ============================================================================

def temporal_train_test_split(data, train_ratio=0.8):
    """
    Realiza un split temporal de los datos (sin shuffle).

    Args:
        data (pd.DataFrame): Dataset completo
        train_ratio (float): Proporción de datos para entrenamiento

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    print_section_header("3. DIVISIÓN TRAIN/TEST TEMPORAL")

    # Separar features y target
    X = data.drop(['Sales'], axis=1)
    y = data['Sales']

    # Split temporal (sin shuffle para respetar orden temporal)
    split_index = int(len(data) * train_ratio)

    x_train = X.iloc[:split_index].copy()
    x_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    print(f"✅ División completada:")
    print(f"   Train set: {x_train.shape[0]:,} registros ({train_ratio*100:.1f}%)")
    print(f"   Test set:  {x_test.shape[0]:,} registros ({(1-train_ratio)*100:.1f}%)")
    print(f"\n📋 Columnas totales: {x_train.shape[1]}")
    print(f"📋 Features finales esperadas: {len(FEATURES)}")

    # Estadísticas de Sales
    print(f"\n📊 Estadísticas de Sales:")
    print(f"   Train - Mean: ${y_train.mean():.2f}, Std: ${y_train.std():.2f}")
    print(f"   Test  - Mean: ${y_test.mean():.2f}, Std: ${y_test.std():.2f}")

    logger.info(f"Split temporal completado - Train: {len(x_train)}, Test: {len(x_test)}")

    return x_train, x_test, y_train, y_test


# ============================================================================
# FUNCIONES DE PIPELINE
# ============================================================================

def build_preprocessing_pipeline(x_train):
    """
    Construye el pipeline de preprocesamiento.

    Args:
        x_train (pd.DataFrame): Datos de entrenamiento

    Returns:
        Pipeline: Pipeline de sklearn configurado
    """
    print_section_header("4. CONSTRUCCIÓN DEL PIPELINE")

    # Identificar features a eliminar
    all_features = set(x_train.columns)
    features_to_keep = set(FEATURES)
    features_to_drop = list(all_features.difference(features_to_keep))

    print(f"✅ Features a mantener: {len(features_to_keep)}")
    print(f"❌ Features a eliminar: {len(features_to_drop)}")

    # Construir pipeline
    print("\n🔧 Construyendo pipeline de preprocesamiento...")

    pipeline = Pipeline([
        # 0. Selección de features
        ('drop_features',
         DropFeatures(features_to_drop=features_to_drop)),

        # 1. Imputación de variables categóricas - Método 'Missing'
        ('cat_missing_imputation',
         CategoricalImputer(
             imputation_method='missing',
             variables=CATEGORICAL_VARS_WITH_NA_MISSING
         )),

        # 2. Imputación de variables categóricas - Método 'Frecuencia'
        ('cat_missing_freq_imputation',
         CategoricalImputer(
             imputation_method='frequent',
             variables=CATEGORICAL_VARS_WITH_NA_FREQUENT
         )),

        # 3. Imputación de variables numéricas - Media
        ('mean_imputation',
         MeanMedianImputer(
             imputation_method='mean',
             variables=NUMERICAL_VARS_WITH_NA
         )),

        # 4. Codificación ordinal (Ship Mode)
        ('quality_mapper',
         ops.Mapper(
             variables=QUAL_VARS,
             mappins=QUAL_MAPPINGS
         )),

        # 5. Frequency Encoding
        ('cat_freq_encode',
         CountFrequencyEncoder(
             encoding_method='count',
             variables=CATEGORICAL_VARS
         )),

        # 6. Transformación logarítmica
        ('continues_log_transform',
         LogTransformer(variables=NUMERICAL_LOG_VARS)),

        # 7. Normalización MinMax (0-1)
        ('Variable_scaler',
         MinMaxScaler())
    ])

    print(f"\n✅ Pipeline construido con {len(pipeline.steps)} pasos:")
    for i, (name, _) in enumerate(pipeline.steps, 1):
        print(f"   {i}. {name}")

    logger.info(f"Pipeline de preprocesamiento construido con {len(pipeline.steps)} pasos")

    return pipeline


def train_pipeline_with_mlflow(pipeline, x_train, y_train):
    """
    Entrena el pipeline y registra métricas en MLflow.

    Args:
        pipeline (Pipeline): Pipeline a entrenar
        x_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento

    Returns:
        Pipeline: Pipeline entrenado
    """
    print_section_header("5. ENTRENAMIENTO DEL PIPELINE")

    with mlflow.start_run(run_name="preprocessing_pipeline"):

        print("🚀 Entrenando pipeline...")
        start_time = datetime.now()

        # Entrenar pipeline
        pipeline.fit(x_train, y_train)

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print(f"✅ Pipeline entrenado en {training_time:.2f} segundos")

        # Registrar parámetros en MLflow
        print("\n📝 Registrando parámetros en MLflow...")
        mlflow.log_param("train_samples", len(x_train))
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("pipeline_steps", len(pipeline.steps))
        mlflow.log_param("imputation_method_categorical", "missing/frequent")
        mlflow.log_param("imputation_method_numerical", "mean")
        mlflow.log_param("encoding_method", "frequency")
        mlflow.log_param("scaling_method", "minmax")

        # Registrar métricas
        print("📊 Registrando métricas en MLflow...")
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("null_percentage",
                         x_train.isnull().sum().sum() / (x_train.shape[0] * x_train.shape[1]))

        # Registrar el pipeline
        print("💾 Registrando pipeline en MLflow...")
        mlflow.sklearn.log_model(
            pipeline,
            "preprocessing_pipeline",
            registered_model_name="stores_sales_preprocessing"
        )

        # Registrar configuración como artefacto
        config_dict = {
            "features": FEATURES,
            "categorical_vars": CATEGORICAL_VARS,
            "numerical_vars_na": NUMERICAL_VARS_WITH_NA,
            "log_transform_vars": NUMERICAL_LOG_VARS,
            "quality_mappings": QUAL_MAPPINGS
        }

        config_path = "pipeline_config.txt"
        with open(config_path, 'w') as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

        mlflow.log_artifact(config_path)
        os.remove(config_path)

        print("\n✅ Pipeline y metadatos registrados en MLflow")
        logger.info(f"Pipeline entrenado y registrado en MLflow - Tiempo: {training_time:.2f}s")

    return pipeline


# ============================================================================
# FUNCIONES DE TRANSFORMACIÓN Y GUARDADO
# ============================================================================

def transform_and_save_data(pipeline, X, y, dataset_name):
    """
    Transforma datos usando el pipeline y guarda el resultado.

    Args:
        pipeline (Pipeline): Pipeline entrenado
        X (pd.DataFrame): Features sin procesar
        y (pd.Series): Variable objetivo
        dataset_name (str): Nombre del dataset (e.g., 'train', 'test')

    Returns:
        pd.DataFrame: Datos transformados
    """
    print(f"\n📄 Procesando {dataset_name}...")

    # Transformar datos
    X_transformed = pipeline.transform(X)

    # Crear DataFrame
    df_transformed = pd.DataFrame(
        data=X_transformed,
        columns=FEATURES
    )

    # Resetear índice de y y concatenar
    y_reset = y.reset_index(drop=True)
    df_transformed = pd.concat(
        [df_transformed, y_reset.rename('Sales')],
        axis=1
    )

    # Guardar archivo
    output_path = f"{DATA_INTERIM_PATH}proc_{dataset_name}.csv"
    df_transformed.to_csv(output_path, index=False)

    print(f"✅ Datos guardados: {output_path}")
    print(f"   Shape: {df_transformed.shape}")

    logger.info(f"Datos {dataset_name} procesados y guardados: {df_transformed.shape}")

    return df_transformed


def validate_processed_data(df_train, df_test):
    """
    Valida los datos procesados.

    Args:
        df_train (pd.DataFrame): Datos de entrenamiento procesados
        df_test (pd.DataFrame): Datos de test procesados
    """
    print_section_header("6. VALIDACIÓN DE DATOS PROCESADOS")

    # 1. Verificar valores nulos
    print("1️⃣  Verificación de valores nulos:")
    train_nulls = df_train.isnull().sum().sum()
    test_nulls = df_test.isnull().sum().sum()

    if train_nulls == 0 and test_nulls == 0:
        print("   ✅ No hay valores nulos en los datos procesados")
    else:
        print(f"   ⚠️  Train nulls: {train_nulls}, Test nulls: {test_nulls}")

    # 2. Verificar dimensiones
    print("\n2️⃣  Verificación de dimensiones:")
    print(f"   Train: {df_train.shape}")
    print(f"   Test:  {df_test.shape}")
    print(f"   Esperadas: {len(FEATURES) + 1} columnas (features + Sales)")

    if df_train.shape[1] == len(FEATURES) + 1:
        print("   ✅ Dimensiones correctas")
    else:
        print("   ⚠️  Dimensiones no coinciden")

    # 3. Verificar rangos (variables normalizadas deben estar en [0,1])
    print("\n3️⃣  Verificación de rangos (MinMax 0-1):")
    numeric_features = df_train.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != 'Sales']

    all_in_range = True
    for col in numeric_features:
        min_val = df_train[col].min()
        max_val = df_train[col].max()

        if min_val < -0.01 or max_val > 1.01:  # Tolerancia pequeña
            print(f"   ⚠️  {col}: [{min_val:.3f}, {max_val:.3f}]")
            all_in_range = False

    if all_in_range:
        print("   ✅ Todas las variables en rango [0, 1]")

    # 4. Estadísticas de Sales
    print("\n4️⃣  Estadísticas de Sales (no normalizada):")
    print(f"   Train - Min: ${df_train['Sales'].min():.2f}, "
          f"Max: ${df_train['Sales'].max():.2f}, "
          f"Mean: ${df_train['Sales'].mean():.2f}")
    print(f"   Test  - Min: ${df_test['Sales'].min():.2f}, "
          f"Max: ${df_test['Sales'].max():.2f}, "
          f"Mean: ${df_test['Sales'].mean():.2f}")

    logger.info("Validación de datos procesados completada")


def save_pipeline(pipeline, filename=PIPELINE_FILENAME):
    """
    Guarda el pipeline entrenado en disco.

    Args:
        pipeline (Pipeline): Pipeline a guardar
        filename (str): Nombre del archivo

    Returns:
        str: Ruta completa del archivo guardado
    """
    print_section_header("7. EXPORTACIÓN DEL PIPELINE")

    pipeline_path = os.path.join(MODELS_PATH, filename)

    joblib.dump(pipeline, pipeline_path)

    file_size_kb = os.path.getsize(pipeline_path) / 1024

    print(f"✅ Pipeline guardado exitosamente")
    print(f"   Ruta: {pipeline_path}")
    print(f"   Tamaño: {file_size_kb:.2f} KB")
    print(f"\n🔧 Para cargar: joblib.load('{pipeline_path}')")

    logger.info(f"Pipeline guardado en: {pipeline_path}")

    return pipeline_path


def print_final_summary(df_train, df_test, pipeline_path):
    """
    Imprime un resumen final del proceso.

    Args:
        df_train (pd.DataFrame): Datos de entrenamiento procesados
        df_test (pd.DataFrame): Datos de test procesados
        pipeline_path (str): Ruta del pipeline guardado
    """
    print_section_header("📊 RESUMEN FINAL")

    print("📁 ARCHIVOS GENERADOS:")
    print(f"   ✓ Pipeline: {pipeline_path}")
    print(f"   ✓ Train data: {DATA_INTERIM_PATH}proc_data_train.csv")
    print(f"   ✓ Test data: {DATA_INTERIM_PATH}proc_data_test.csv")

    print("\n📊 DIMENSIONES:")
    print(f"   ✓ Train: {df_train.shape[0]:,} registros × {df_train.shape[1]} columnas")
    print(f"   ✓ Test:  {df_test.shape[0]:,} registros × {df_test.shape[1]} columnas")

    print(f"\n✅ Features finales ({len(FEATURES)} variables):")
    for i, feat in enumerate(FEATURES, 1):
        print(f"   {i:2d}. {feat}")

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE".center(70))
    print("=" * 70)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de feature engineering.
    """
    try:
        # Configuración inicial
        print_section_header("🚀 FEATURE ENGINEERING Y PREPROCESAMIENTO")
        logger.info("="*70)
        logger.info("Iniciando proceso de Feature Engineering")
        logger.info("="*70)

        # Crear directorios necesarios
        create_directories()

        # Configurar MLflow
        setup_mlflow()

        # 1. Cargar y preparar datos
        data = load_and_prepare_data()

        # 2. Crear valores nulos artificiales (para demostración)
        data = create_artificial_nulls(data)

        # 3. Split temporal
        x_train, x_test, y_train, y_test = temporal_train_test_split(data)

        # 4. Construir pipeline
        pipeline = build_preprocessing_pipeline(x_train)

        # 5. Entrenar pipeline con tracking de MLflow
        pipeline = train_pipeline_with_mlflow(pipeline, x_train, y_train)

        # 6. Transformar y guardar datos procesados
        print_section_header("5. TRANSFORMACIÓN Y GUARDADO DE DATOS")
        df_train = transform_and_save_data(pipeline, x_train, y_train, "data_train")
        df_test = transform_and_save_data(pipeline, x_test, y_test, "data_test")

        # 7. Validar datos procesados
        validate_processed_data(df_train, df_test)

        # 8. Guardar pipeline
        pipeline_path = save_pipeline(pipeline)

        # 9. Resumen final
        print_final_summary(df_train, df_test, pipeline_path)

        logger.info("Proceso de Feature Engineering completado exitosamente")

        return 0

    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())