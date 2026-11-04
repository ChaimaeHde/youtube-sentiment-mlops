import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def get_logger(name: str = "sentiment_api") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:                      # évite les doublons au rechargement
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        LOG_DIR.mkdir(exist_ok=True)
        file = logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")
        file.setFormatter(fmt)
        logger.addHandler(file)
    except OSError:
        pass                                 # en conteneur, la console suffit
    return logger