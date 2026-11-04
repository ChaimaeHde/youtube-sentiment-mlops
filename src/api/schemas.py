from pydantic import BaseModel, Field, field_validator

MAX_BATCH = 500


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                                examples=[["I love this video!", "Worst content ever"]])

    @field_validator("comments")
    @classmethod
    def check_not_all_empty(cls, v: list[str]) -> list[str]:
        if not any(c.strip() for c in v):
            raise ValueError("Tous les commentaires sont vides")
        return v


class Prediction(BaseModel):
    text: str
    sentiment: str          # "positive" | "neutral" | "negative"
    label: int              # 1 | 0 | -1
    confidence: float       # probabilité de la classe prédite


class Stats(BaseModel):
    total: int
    positive: int
    neutral: int
    negative: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    avg_confidence: float


class PredictBatchResponse(BaseModel):
    predictions: list[Prediction]
    stats: Stats
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str