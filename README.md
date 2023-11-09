# archi-simple
simple architecture pour Dockeriser 2 modèles de ML. Les étapes pour commencer

## Step 0: creating and activate venv (Optionnel)

Peut servir d'environnement de test pour s'assurer tout marche bien. Mais vous pouvez également créer 2 environnements virtuels associé à chaque modèle (ce qui est recommandé pour des omdèles lourds).

```sh
# creation sous windows
python -m venv venv

# activation sous windows
.\venv\Scripts\activate
```

La création du `venv` à la racine c'est pour installer un kernel de jupyter notebook, mais vous pouvez créer 2 environnements virtuels dans les dossiers de chacun des modèles.

## Step 1: Train the models

```sh
cd .\src\dt_app\
python .\train_dt.py
cd ..
cd .\linear_regression_app\
python .\train_LR.py
```

Pour avoir une idée des valeurs
```sh
#example of X: DT
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
]

# exemple X LR:
[[1.64349073e+00]
 [1.84562169e+00]
 [1.59503241e+00]
 [1.64345363e+00]
 [9.27661209e-01]
]
```
## Step 3: Build and test separated APIs

## Step 4: Dock the whole thing

## Step 5: create and run the docker compose

```sh
# LR
curl -X 'POST' \
  'http://localhost:8001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [
   5.1, 3.5,1.4,0.2
  ]
}'

# DT
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [
    1
  ]
}'
```
Vous pouvez également tester via la librairie requests (voir dans ./notebooks/client.ipynb).
