import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.predictor import SentimentPredictor
from src.api.schemas import HealthResponse, PredictBatchRequest, PredictBatchResponse
from src.utils.logger import get_logger

VERSION = "1.0.0"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
logger = get_logger()
predictor: SentimentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle UNE SEULE FOIS au démarrage."""
    global predictor
    try:
        predictor = SentimentPredictor(MODEL_DIR)
        logger.info("Modèle chargé depuis %s", MODEL_DIR)
    except Exception as e:
        logger.error("Échec du chargement du modèle : %s", e)
        predictor = None
    yield
    logger.info("Arrêt de l'API")


app = FastAPI(title="YouTube Sentiment API", version=VERSION, lifespan=lifespan,
              description="Analyse de sentiment de commentaires (TF-IDF + LogReg)")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https://www\.youtube\.com)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "YouTube Sentiment API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return HealthResponse(status="ok", model_loaded=True, version=VERSION)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    t0 = time.perf_counter()
    try:
        results = predictor.predict(request.comments)
        stats = predictor.compute_stats(results)
    except Exception as e:
        logger.exception("Erreur de prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")
    ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Batch de %d commentaires traité en %.2f ms", len(results), ms)
    return PredictBatchResponse(predictions=results, stats=stats, processing_time_ms=ms)