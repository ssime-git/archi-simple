import requests
import streamlit as st

def lr_api_call(features):
    url = "http://linear_regression_service:8000/linear_predict"
    headers = {'accept': 'application/json'}
    data = {"features": features}

    response = requests.post(url, json=data, headers=headers)
    return response

def dt_api_call(features):
    url = "http://decision_tree_service:8001/dt_predict"
    headers = {'accept': 'application/json'}
    data = {"features": features}

    response = requests.post(url, json=data, headers=headers)
    return response

def main():
    st.title("Mon application Streamlit")
    st.write("Bienvenue sur mon application !")

    user_query = st.text_input("Entrez vos caractéristiques", "")

    if st.button("Prédiction Régression Linéaire"):
        features = [float(x) for x in user_query.split(',')]  # Assuming features are comma-separated
        response = lr_api_call(features)
        if response.status_code == 200:
            st.write("Prédiction Régression Linéaire:", response.json())
        else:
            st.error(f"Erreur Régression Linéaire: Status {response.status_code}")

    if st.button("Prédiction Arbre de Décision"):
        features = [float(x) for x in user_query.split(',')]  # Assuming features are comma-separated
        response = dt_api_call(features)
        if response.status_code == 200:
            st.write("Prédiction Arbre de Décision:", response.json())
        else:
            st.error(f"Erreur Arbre de Décision: Status {response.status_code}")

if __name__ == "__main__":
    main()