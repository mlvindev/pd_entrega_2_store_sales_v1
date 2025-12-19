import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# Clase para codificacion de variables categoricas ordinales
class Mapper(BaseEstimator, TransformerMixin):

    # Implementación de Constructor
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError("Las variables deben de ser tipo lista")
        
        self.variables = variables
        self.mappings = mappings

    # Implementación de Método fit
    def fit(self, X, y=None):
        # fit no hace nada, pero es requisito del pipeline
        return self
    
    # Implementación de Método .transform
    def transform(self, X):
         X = X.copy()
         for variable in self.variables:
             X[variable] = X[variable].map(self.mappings)
         return X

        
# Clase para manejo de variables temporales en el modelo de House Price
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):

    # Implementación de Constructor
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError("Las variables deben estar incluida en una lista")
        
        self.variables = variables
        self.reference_variable = reference_variable

    # Implementación de Método fit
    def fit(self, X, y=None):
        # fit no hace nada, pero es requisito del pipeline
        return self
    
    # Implementación de Método .transform
    def transform(self, X):
         X = X.copy()
         for feauture in self.variables:
             X[feauture] = X(self.reference_variable) - X(feauture)
         return X