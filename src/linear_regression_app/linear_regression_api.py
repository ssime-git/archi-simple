from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle de régression linéaire pré-entraîné
model = joblib.load('linear_regression_model.pkl')

# Initialiser FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la structure des données de demande
class RequestData(BaseModel):
    features: list

# Définir une route pour faire des prédictions
@app.post("/predict")
async def predict(data: RequestData):
    # Convertir les données en un array numpy pour la prédiction
    features = np.array(data.features).reshape(-1, 1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
