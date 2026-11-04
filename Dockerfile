# Image légère, même version de Python que l'entraînement
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# 1. Dépendances d'abord : couche mise en cache tant que les requirements ne changent pas
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Uniquement les fichiers nécessaires à l'API
COPY src/__init__.py src/__init__.py
COPY src/utils/ src/utils/
COPY src/api/ src/api/
COPY models/sentiment_model.joblib models/tfidf_vectorizer.joblib models/
COPY app_api.py .

# 3. Exécution avec un utilisateur non-root (sécurité)
RUN useradd -m appuser
USER appuser

EXPOSE 7860
CMD ["python", "app_api.py"]