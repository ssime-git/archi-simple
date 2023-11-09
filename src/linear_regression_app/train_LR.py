# Étape 1: Importer les bibliothèques nécessaires
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Étape 2: Créer des données pour l'entrainement
# Générer des données d'exemple
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Étape 3: Entraîner le modèle de régression linéaire
# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instancier et entraîner le modèle de régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Étape 4: Sauvegarder le modèle entraîné
joblib.dump(lin_reg, 'linear_regression_model.pkl')

print(X)