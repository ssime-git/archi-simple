from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Charger le modèle de classificateur d'arbre de décision pré-entraîné
#model = joblib.load('decision_tree_classifier.pkl')

# Initialiser FastAPI
app = FastAPI()

# Définir un modèle Pydantic pour la structure des données de demande
class RequestData(BaseModel):
    features: list

# Utilisez une fonction pour obtenir le chemin du modèle
def get_model_path():
    return f"{os.path.dirname(os.path.realpath(__file__))}/decision_tree_classifier.pkl" #"decision_tree_classifier.pkl"

# Créez une fonction pour charger le modèle
def load_model():
    model_path = get_model_path()
    model = joblib.load(model_path)
    return model

# Définir une route pour faire des prédictions
@app.post("/predict")
async def predict(data: RequestData):
    model = load_model()
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}

