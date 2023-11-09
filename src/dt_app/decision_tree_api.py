from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Charger le modèle de classificateur d'arbre de décision pré-entraîné
model = joblib.load('decision_tree_classifier.pkl')

# Initialiser FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la structure des données de demande
class RequestData(BaseModel):
    features: list

# Définir une route pour faire des prédictions
@app.post("/predict")
async def predict(data: RequestData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
