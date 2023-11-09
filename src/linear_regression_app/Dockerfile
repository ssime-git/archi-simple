# Utiliser une image de base Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du modèle et de l'API dans le conteneur
COPY linear_regression_model.pkl ./
COPY linear_regression_api.py ./

# Exposer le port sur lequel FastAPI va s'exécuter
EXPOSE 80

# Exécuter l'API avec Uvicorn
CMD ["uvicorn", "linear_regression_api:app", "--host", "0.0.0.0", "--port", "80"]
