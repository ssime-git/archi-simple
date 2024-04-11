# archi-simple
Very basic simpel architecture with :
- 2 ML models
- 2 diferent API running in docker
- 1 front end streamlit app to consume the model

## Step 0: creating and activate venv (Optionnel)

Can be used as a test environment to ensure everything works well. However, you can also create 2 virtual environments associated with each model (which is recommended for heavy models).

```sh
# creation sous windows
python -m venv venv

# activation sous windows
.\venv\Scripts\activate
```

Creating the `venv` at the root is for installing a Jupyter notebook kernel, but you can create 2 virtual environments in the folders of each of the models.

## Step 1: Train the model

Important : Make sure to train the models before running the test

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
docker compose up --build
```

## Test
Use the client notebook: ./notebooks/client.ipynb

## inspiration pour écrire les test

- https://github.com/pamelafox/scikitlearn-model-to-fastapi-app
