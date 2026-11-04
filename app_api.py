"""Point d'entrée de production : lance l'API sur le port défini par la variable PORT."""
import os

import uvicorn

from src.api.main import app  # même code qu'en local : aucune divergence

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")),
                workers=1, log_level="info")