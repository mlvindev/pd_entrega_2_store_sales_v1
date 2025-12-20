"""
Script: Inference Calculation
Descripción: Cálculo de predicciones y métricas de evaluación del modelo
Autor: Grupo6

CORRECCIÓN: Este script NO crea features derivadas manualmente.
El pipeline se encarga de todas las transformaciones.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os
import logging
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# 2. CONSTANTES
# ============================================================================

# Configuración de rutas absolutas teniendo dvc
DATA_PATH = 'data/raw/stores_sales_forecasting_updated_v3.1.csv'
PIPELINE_PATH = 'models/stores_sales_forecasting_pipeline.pkl'
RESULTS_DIR = 'results/'

LOG_FILE = 'inference_system.log'
TEST_SIZE = 0.2
DATE_FORMAT = "%Y%m%d_%H%M%S"
OUTPUT_FILE_PREFIX = 'predicciones_'


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
logger.info("Iniciando proceso de inferencia")
logger.info("=" * 80)


# ============================================================================
# 4. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("CARGA Y PREPARACIÓN DE DATOS")
print("=" * 80 + "\n")

# Cargar dataset
inference_data = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')

print(f"✅ Dataset cargado: {inference_data.shape}")
logger.info("Dataset cargado: %s", inference_data.shape)

# IMPORTANTE: NO crear features derivadas manualmente
# El pipeline se encarga de todas las transformaciones
print("ℹ️  El pipeline manejará todas las transformaciones de features")
logger.info("Pipeline manejará la creación de features derivadas")


# ============================================================================
# 5. DIVISIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("DIVISIÓN DE DATOS")
print("=" * 80 + "\n")

# Separar features y target
# El pipeline espera recibir el DataFrame completo SIN Sales
X = inference_data.drop(['Sales'], axis=1).copy()
y = inference_data['Sales'].copy()

# Split temporal (80/20) - SIN shuffle para mantener orden temporal
split_index = int(len(inference_data) * (1 - TEST_SIZE))

x_train = X.iloc[:split_index].copy()
x_test = X.iloc[split_index:].copy()
y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print(f"📊 Datos divididos:")
print(f"   Train: {len(x_train):,} registros ({len(x_train)/len(X)*100:.1f}%)")
print(f"   Test:  {len(x_test):,} registros ({len(x_test)/len(X)*100:.1f}%)")
print(f"   Columnas en X_test: {x_test.shape[1]}")

logger.info("Train set: %d registros", len(x_train))
logger.info("Test set: %d registros", len(x_test))
logger.info("Número de features: %d", x_test.shape[1])


# ============================================================================
# 6. CARGA DEL PIPELINE
# ============================================================================
print("\n" + "=" * 80)
print("CARGA DEL PIPELINE")
print("=" * 80 + "\n")

stores_sales_forecasting_pipeline = joblib.load(PIPELINE_PATH)

print(f"✅ Pipeline cargado desde: {PIPELINE_PATH}")
print(f"   Número de pasos en el pipeline: {len(stores_sales_forecasting_pipeline.steps)}")

# Mostrar pasos del pipeline para debugging
print("   Pasos del pipeline:")
for i, (name, _) in enumerate(stores_sales_forecasting_pipeline.steps, 1):
    print(f"      {i}. {name}")

logger.info("Pipeline cargado: %s", PIPELINE_PATH)


# ============================================================================
# 7. GENERACIÓN DE PREDICCIONES
# ============================================================================
print("\n" + "=" * 80)
print("GENERACIÓN DE PREDICCIONES")
print("=" * 80 + "\n")

start_time = datetime.now()

try:
    predicciones = stores_sales_forecasting_pipeline.predict(x_test)
    end_time = datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    
    print("✅ Predicciones generadas exitosamente")
    print(f"   Tiempo de inferencia: {inference_time:.2f} segundos")
    print(f"   Número de predicciones: {len(predicciones)}")
    print(f"   Tiempo por predicción: {inference_time/len(predicciones)*1000:.2f} ms")
    
    logger.info("Predicciones generadas: %d", len(predicciones))
    logger.info("Tiempo de inferencia: %.2f segundos", inference_time)
    
except Exception as e:
    print(f"❌ Error generando predicciones: {e}")
    logger.error("Error generando predicciones: %s", str(e))
    raise


# ============================================================================
# 8. CÁLCULO DE MÉTRICAS
# ============================================================================
print("\n" + "=" * 80)
print("EVALUACIÓN DE PREDICCIONES")
print("=" * 80 + "\n")

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)
mape = np.mean(np.abs((y_test - predicciones) / y_test)) * 100

print("MÉTRICAS DE EVALUACIÓN:")
print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"   MAE (Mean Absolute Error):      {mae:.2f}")
print(f"   R² Score:                       {r2:.4f}")
print(f"   MAPE (Mean Absolute % Error):   {mape:.2f}%")

logger.info("RMSE: %.2f", rmse)
logger.info("MAE: %.2f", mae)
logger.info("R² Score: %.4f", r2)
logger.info("MAPE: %.2f%%", mape)

# Estadísticas descriptivas de predicciones
print("\nESTADÍSTICAS DE PREDICCIONES:")
print(f"   Mínimo:  {predicciones.min():,.2f}")
print(f"   Máximo:  {predicciones.max():,.2f}")
print(f"   Media:   {predicciones.mean():,.2f}")
print(f"   Mediana: {np.median(predicciones):,.2f}")
print(f"   Std Dev: {predicciones.std():,.2f}")

# Estadísticas descriptivas de valores reales
print("\nESTADÍSTICAS DE VALORES REALES:")
print(f"   Mínimo:  {y_test.min():,.2f}")
print(f"   Máximo:  {y_test.max():,.2f}")
print(f"   Media:   {y_test.mean():,.2f}")
print(f"   Mediana: {y_test.median():,.2f}")
print(f"   Std Dev: {y_test.std():,.2f}")


# ============================================================================
# 9. CREACIÓN DE DATAFRAME CON RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("GENERACIÓN DE REPORTE DE PREDICCIONES")
print("=" * 80 + "\n")

results_df = pd.DataFrame({
    'Index': y_test.index,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': predicciones,
    'Absolute_Error': np.abs(y_test.values - predicciones),
    'Percentage_Error': np.abs((y_test.values - predicciones) / y_test.values) * 100,
    'Residual': y_test.values - predicciones
})

# Incluir columnas originales si existen
if 'Order Date' in x_test.columns:
    results_df['Order_Date'] = x_test['Order Date'].values

if 'Category' in x_test.columns:
    results_df['Category'] = x_test['Category'].values

if 'Region' in x_test.columns:
    results_df['Region'] = x_test['Region'].values

print("PRIMERAS 10 PREDICCIONES:")
print(results_df[
    ['Actual_Sales', 'Predicted_Sales', 'Absolute_Error', 'Percentage_Error']
].head(10).to_string(index=False))


# ============================================================================
# 10. GUARDADO DE RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("GUARDADO DE RESULTADOS")
print("=" * 80 + "\n")

# Crear directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# Generar timestamp y nombre del archivo
timestamp = datetime.now().strftime(DATE_FORMAT)
predictions_file = f'{RESULTS_DIR}{OUTPUT_FILE_PREFIX}{timestamp}.csv'

# Guardar resultados
results_df.to_csv(predictions_file, index=False)

print(f"✅ Predicciones guardadas en: {predictions_file}")
print(f"   Total de registros: {len(results_df):,}")
print(f"   Columnas guardadas: {len(results_df.columns)}")

logger.info("Resultados guardados en: %s", predictions_file)
logger.info("Total de registros guardados: %d", len(results_df))


# ============================================================================
# 11. RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 100)
print(" " * 35 + "RESUMEN FINAL")
print("=" * 100 + "\n")

print("📊 MÉTRICAS DEL MODELO:")
print(f"   RMSE:  {rmse:.2f}")
print(f"   MAE:   {mae:.2f}")
print(f"   R²:    {r2:.4f}")
print(f"   MAPE:  {mape:.2f}%")

print("\n⏱️  RENDIMIENTO:")
print(f"   Tiempo total de inferencia: {inference_time:.2f} segundos")
print(f"   Predicciones generadas:     {len(predicciones):,}")
print(f"   Velocidad:                  {len(predicciones)/inference_time:.2f} pred/seg")

print("\n📁 ARTEFACTOS GENERADOS:")
print(f"   ✓ Archivo de predicciones: {predictions_file}")
print(f"   ✓ Log del sistema:         {LOG_FILE}")

print("\n📈 DISTRIBUCIÓN DE ERRORES:")
error_percentiles = np.percentile(results_df['Absolute_Error'], [25, 50, 75, 90, 95])
print(f"   P25:  ${error_percentiles[0]:.2f}")
print(f"   P50:  ${error_percentiles[1]:.2f}")
print(f"   P75:  ${error_percentiles[2]:.2f}")
print(f"   P90:  ${error_percentiles[3]:.2f}")
print(f"   P95:  ${error_percentiles[4]:.2f}")

# Análisis de predicciones con mayor error
top_errors = results_df.nlargest(5, 'Absolute_Error')[
    ['Actual_Sales', 'Predicted_Sales', 'Absolute_Error', 'Percentage_Error']
]
print("\n⚠️  TOP 5 PREDICCIONES CON MAYOR ERROR:")
print(top_errors.to_string(index=False))

print("\n" + "=" * 100)
print("✅ PROCESO DE INFERENCIA COMPLETADO EXITOSAMENTE")
print("=" * 100 + "\n")

logger.info("=" * 80)
logger.info("Proceso de inferencia completado exitosamente")
logger.info("RMSE: %.2f, R²: %.4f", rmse, r2)
logger.info("Archivo generado: %s", predictions_file)
logger.info("=" * 80)