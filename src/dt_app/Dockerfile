# Utiliser une image de base Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du modèle et de l'API dans le conteneur
COPY decision_tree_classifier.pkl ./
COPY decision_tree_api.py ./

# Exposer le port sur lequel FastAPI va s'exécuter
EXPOSE 80

# Exécuter l'API avec Uvicorn
CMD ["uvicorn", "decision_tree_api:app", "--host", "0.0.0.0", "--port", "80"]
