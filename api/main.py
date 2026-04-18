import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

try:
    from .database import engine, get_db
    from .model import predict_histopathology
    from .models_db import Base, Prediction
    from .utils import load_image_from_bytes
except ImportError:
    from database import engine, get_db
    from model import predict_histopathology
    from models_db import Base, Prediction
    from utils import load_image_from_bytes


LOGGER = logging.getLogger("oral_histopathology_api")

try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:
    LOGGER.warning("No se pudo inicializar la tabla de predicciones: %s", exc)

app = FastAPI(
    title="Oral Histopathology Classifier API",
    description="API para clasificacion de micrografias histopatologicas (Normal vs OSCC)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "project": "app_prediccion_cancer_bucal_histopatologico",
        "architecture": "EfficientNet-B0 + custom dense head",
        "classes": ["Normal", "OSCC"],
        "input_size": "224x224",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen valida")

    try:
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
        result = predict_histopathology(image)

        db_prediction = Prediction(
            project="app_prediccion_cancer_bucal_histopatologico",
            image_name=str(file.filename or "uploaded_image"),
            predicted_label=str(result.get("label", "Unknown")),
            confidence=float(result.get("confidence", 0.0)),
            probabilities=result,
            is_correct=None,
        )
        try:
            db.add(db_prediction)
            db.commit()
        except Exception as db_exc:
            db.rollback()
            LOGGER.warning("No se pudo guardar prediccion en DB: %s", db_exc)

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {exc}") from exc
