from fastapi import FastAPI, UploadFile, File, HTTPException
from src.inference.predict import predict_image

app = FastAPI(title="OsteoLab", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": "0.1.0",
        "model_source": predict_image.__module__,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return predict_image(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))