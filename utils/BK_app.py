"""
Script web services
=========================================================

Este módulo maneja la construccion del api para el preprocesamiento de datos del proyecto Store Sales Forecasting.

Author: Grupo6
"""

from flask import Flask
from flask import request
#from flask import jsonify
#from datetime import datetime
import pandas as pd
import joblib

# Inicializar flask
app = Flask(__name__)

# Se llama al modelo creado en /models
modelo_hpp = joblib.load("../models/stores_sales_forecasting_pipeline.pkl")

# Metodo POST
@app.route("/predict_single", methods=['POST'])
def predict_single():
    """
    Definicion de funcion para predicciones
    Author: Grupo6
    """
    data = request.get_json()
    df_data = pd.DataFrame(data)
    prediction = modelo_hpp.predict(df_data)
    print(prediction)
    return "Hola"

# Metodo GET - Hola mundo
# @app.route("/saludar", methods=['GET'])
# def saludo_v1():
#    return "Hola a todos"

# Metodo GET con parametros
# @app.route("/saludar_v2/<nombre>", methods=['GET'])
# def saludo_v2(nombre):
#    return "Hola {nombre}"

# Metodo GET con parametros numericos para sumar
# @app.route("/suma", methods=['GET'])
# def sumar():
#    x = request.args.get("x", type=int)
#    y = request.args.get("y", type=int)
#    return f"La suma de los valores es: {x+y}"

# Metodo GET con parametros numericos para restar
# @app.route("/resta", methods=['POST'])
# def restar_numeros():
#    data = request.get_json()

#   x = data.get["x"]
#    y = data.get["y"]

#    output = jsonify({"x": x, "y": y, "resultado": x - y})
#    return output


# Correr main
if __name__ == "__main__":
    app.run(debug=True)
