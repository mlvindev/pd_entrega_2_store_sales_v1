import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin

#clase para codificacion de variables categoricas ordinales
class Mapper(BaseEstimator, TransformerMixin):

    #constructor
    def __init__(self, variables, mappins):
        if not isinstance(variables, list):
            raise ValueError("Las Variables deben de ser tipo lista")
        

        self.variables = variables
        self.mappins = mappins

    #metodo fit
    def fit(self, X, y=None):
        #fit no hace nada, pero es requisito del pipeline
        return self
    

    def transform(self, X):
        X=X.copy()
        for variable in self.variables:
            X[variable]=X[variable].map(self.mappins)
        return X
    

#Clase para manejo de variables temporales en el modelo de House Price

class TremporalVariableTransformer(BaseEstimator, TransformerMixin):

 #Constructor

    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError("Las varibles debe ser incluida en una lista.")

        self.variables = variables
        self.reference_variable = reference_variable

 #metodo fit para habilitar metodo transform

    def fit(self, X, y=None):
        return self

 #metodo para transformar variables temporales.

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
            return X