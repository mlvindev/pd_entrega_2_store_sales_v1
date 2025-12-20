"""
Módulo para la creación de features de ingeniería.

Este módulo contiene funciones para transformar y crear nuevas características
a partir de los datos originales del dataset.
"""

#import os
#import sys
import logging
import pandas as pd
#import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.transformation import LogTransformer
from feature_engine.selection import DropFeatures
#import operators
import operators as ops

logging.basicConfig(filename="prod_ml_system.log",
                    encoding="utf-8",
                    filemode="a",
                    level=logging.INFO,
                    format="{asctime},{levelname},{message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M")

# Variables globales
#imputacion de variables categoricas con imputación por frecuencia
CATEGORICAL_VARS_WITH_NA_FREQUENT=['Sub-Category']

#Imputacion de variables númericas con imputacion por media
NUMERICAL_VARS_WITH_NA=['Quantity','Discount']

# Imputación de variables categóricas con valor faltante (Missing)
CATEGORICAL_VARS_WITH_NA_MISSING = ['Segment']

# Variables a eliminar
DROP_FEATURES = ['Row ID',
                 'Order ID',
                 'Customer ID',
                 'Customer Name',
                 'Order Date',
                 'Ship Date',
                 'Branch',
                 'Postal Code',
                 'Product ID',
                 'Product Name']

# Variables para transformación logarítmica
NUMERICAL_LOG_VARS = ['Quantity']

# Variables para codificación ordinal (calidad)
QUAL_VARS = ['Ship Mode']

# Variables para codificación por frecuencia (no ordinal)
CATEGORICAL_VARS = ['Segment', 'Sub-Category', 'Country', 'City','State' , 'Region', 'Category']

# Mapeos para variables categóricas de calidad
QUAL_MAPPINGS = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3}

# Variables numéricas principales
NUMERICAL_VARS = ['Quantity',
                  'Discount',
                  'Profit',
                  'Order_Month',
                  'Order_Quarter',
                  'Days to Ship']

# Variables finales para el modelo
FEATURES = ['Quantity',
            'Discount',
            'Profit',
            'Ship Mode',
            'Segment',
            'Country',
            'City',
            'State',
            'Region',
            'Category',
            'Sub-Category',
            'Order_Month',
            'Order_Quarter',
            'Days to Ship']

def load_n_pre_data():
    """
    Carga y preprocesa datos de ventas, generando features y realizando split de entrenamiento.
    
    Returns:
        tuple: (x_train, y_train) - Features y target de entrenamiento (80% de datos).
    
    Notes:
        - Convierte fechas y crea variables derivadas (mes, trimestre, días de envío).
        - Split temporal 80/20.
    """
    # Cargar el dataset
    data_train= pd.read_csv('../data/raw/stores_sales_forecasting_updated_v3.1.csv',
                            sep=';',
                            encoding='utf-8')

    # Convertir fechas
    data_train['Order Date'] = pd.to_datetime(
        data_train['Order Date'],
        dayfirst=True,
        errors='coerce')
    data_train['Ship Date'] = pd.to_datetime(
        data_train['Ship Date'],
        dayfirst=True,
        errors='coerce')

    # Variables derivadas
    data_train['Order_Month'] = data_train['Order Date'].dt.month
    data_train['Order_Quarter'] = data_train['Order Date'].dt.quarter
    data_train['Days to Ship'] = (data_train['Ship Date'] - data_train['Order Date']).dt.days

    # Asegurar que sea numérico para evitar errores
    data_train['Postal Code'] = pd.to_numeric(data_train['Postal Code'], errors='coerce')
    data_train['Discount'] = pd.to_numeric(data_train['Discount'], errors='coerce')
    data_train['Quantity'] = pd.to_numeric(data_train['Quantity'], errors='coerce')
    data_train['Profit'] = pd.to_numeric(data_train['Profit'], errors='coerce')
    data_train['Order_Month'] = pd.to_numeric(data_train['Order_Month'], errors='coerce')
    data_train['Order_Quarter'] = pd.to_numeric(data_train['Order_Quarter'], errors='coerce')
    data_train['Days to Ship'] = pd.to_numeric(data_train['Days to Ship'], errors='coerce')

    # Split en train test
    X = data_train.drop(['Sales'], axis=1)
    y = data_train['Sales']
    split_index = int(len(data_train) * 0.8)
    x_train = X.iloc[:split_index].copy()
    #x_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    #y_test = y.iloc[split_index:].copy()
    return x_train, y_train

def create_n_config_preproc_pipeline(x_train, y_train):
    """
    Crea y entrena un pipeline de preprocesamiento de datos de ventas.
    
    Aplica: selección de features, imputación, codificación, transformación log
    y normalización. Guarda el pipeline en '../models/'.
    
    Args:
        x_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Variable objetivo.
    
    Returns:
        Pipeline: Pipeline de sklearn entrenado.
    """
    all_features = set(
        x_train.columns)
    features_to_drop = all_features.difference(FEATURES)
    features_to_drop = list(
        features_to_drop)
    stores_sales_forecasting_data_pre_proc=Pipeline([
        #0.Seleccion de features para el modelo
        ('drop_features', DropFeatures(features_to_drop=features_to_drop)),
        #1.Imputacion de variables categoricas
        ('cat_missing_imputation', CategoricalImputer(
            imputation_method='missing',
            variables=CATEGORICAL_VARS_WITH_NA_MISSING)),
        #2.Imputacion de variables categoricas por frecuencia
        ('cat_missing_freq_imputation', CategoricalImputer(
            imputation_method='frequent',
            variables=CATEGORICAL_VARS_WITH_NA_FREQUENT)),
        #3.Imputacion de variables númericas
        ('mean_imputation', MeanMedianImputer(imputation_method='mean',
                                              variables=NUMERICAL_VARS_WITH_NA)),
        #4.Codificacion de variables categoricas
        ('quality_mapper', ops.Mapper(variables=QUAL_VARS,
                                            mappins=QUAL_MAPPINGS)),
        #5.Codificacion por Frecuency encoding
        ('cat_freq_encode', CountFrequencyEncoder(encoding_method='count',
                                                  variables=CATEGORICAL_VARS)),
        #6.Transformacion de variables continuas
        ('continues_log_transform', LogTransformer(variables=NUMERICAL_LOG_VARS)),
        #7.Normalizacion de variables
        ('Variable_scaler', MinMaxScaler())
    ])

    stores_sales_forecasting_data_pre_proc.fit(x_train, y_train)
    joblib.dump(stores_sales_forecasting_data_pre_proc,
                '../models/stores_sales_forecasting_data_pre_proc.pkl')

    return stores_sales_forecasting_data_pre_proc

def save_procesed_data(x,y, str_df_name, stores_sales_forecasting_data_pre_proc):
    """
    Guarda los datos procesados en un archivo.
    
    Args:
        x (pd.DataFrame): Features transformadas.
        y (pd.Series): Variable objetivo.
        str_df_name (str): Nombre del archivo de salida.
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados.
    """
    X_transformed = stores_sales_forecasting_data_pre_proc.transform(x)
    df_X_train_transformed= pd.DataFrame(data=X_transformed, columns=FEATURES)
    y = y.reset_index()
    df_transformed= pd.concat([df_X_train_transformed, y['Sales']], axis=1)
    df_transformed.to_csv(f"../data/interim/proc_{str_df_name}.csv", index=False)

def main():
    try:
        logging.info("Iniciando Preprocesamiento de Datos")
        print("Iniciando Preprocesamiento de Datos")
        # Se carga y configuran los datos de entrada
        x_train, y_train = load_n_pre_data()
        logging.info("Datos cargados y configurados correctamente")
        print("Datos cargados y configurados correctamente")
        # Se crea y configura el pipeline
        pipeline = create_n_config_preproc_pipeline(x_train, y_train)
        logging.info("Datos cargados y configurados correctamente")
        print("Datos cargados y configurados correctamente")
        # Se guardan los datos pre-procesados para entrenamiento
        save_procesed_data(x_train, y_train, "data_train", pipeline)
        logging.info("Datos de entrenamiento guardados correctamente")
        print("Datos de entrenamiento guardados correctamente")
    except Exception as ex:
        #logging.error(f"Error {ex}")
        logging.error("Error inesperado: %s", ex)
        print(f"Error - {ex}")

if __name__ == "__main__":
    main()
