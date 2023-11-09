# Étape 1: Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle

# Étape 2: Charger les données Iris pour l'entrainement
iris = load_iris()
X, y = iris.data, iris.target

# Étape 3: Entraîner le modèle d'arbre de décision
# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instancier et entraîner le classificateur d'arbre de décision
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

# Étape 4: Sauvegarder le modèle entraîné
#joblib.dump(tree_clf, 'decision_tree_classifier.pkl') # few issues when loading

with open('decision_tree_classifier.pkl', 'wb') as f:
    pickle.dump(tree_clf, f)

print(X)