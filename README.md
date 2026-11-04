# Analyse de sentiment des commentaires YouTube

Système MLOps complet : modèle de classification (TF-IDF + Logistic Regression),
API FastAPI déployée sur Hugging Face Spaces et extension Chrome.

## Structure du projet

data/              données brutes et traitées
models/            modèle et vectoriseur (.joblib)
src/data/          téléchargement, nettoyage, EDA
src/models/        entraînement et évaluation
src/api/           API FastAPI
src/utils/         fonctions partagées
chrome-extension/  interface utilisateur

## Installation

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Pipeline de données

python -m src.data.download_data
python -m src.data.clean_data
python -m src.data.eda

## Entraînement

python -m src.models.train
