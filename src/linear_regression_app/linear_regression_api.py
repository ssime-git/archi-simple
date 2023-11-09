from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Charger le modèle de régression linéaire pré-entraîné
#model = joblib.load('linear_regression_model.pkl')

# Initialiser FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la structure des données de demande
class RequestData(BaseModel):
    features: list

# Utilisez une fonction pour obtenir le chemin du modèle
def get_model_path():
    return f"{os.path.dirname(os.path.realpath(__file__))}/linear_regression_model.pkl" #"linear_regression_model.pkl"

# Créez une fonction pour charger le modèle
def load_model():
    model_path = get_model_path()
    model = joblib.load(model_path)
    return model

# Définir une route pour faire des prédictions
@app.post("/predict")
async def predict(data: RequestData):
    model = load_model()
    # Convertir les données en un array numpy pour la prédiction
    features = np.array(data.features).reshape(-1, 1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
