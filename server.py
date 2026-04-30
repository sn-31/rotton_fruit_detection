"""
FruitSense — Inference API Server
Serves the trained Keras model over HTTP AND hosts the frontend HTML.

Install deps:
    pip install fastapi uvicorn pillow numpy tensorflow

Run:
    python server.py
    -> Opens http://localhost:8000 automatically in your browser
"""

import io
import json
import webbrowser
import threading
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("fruit_model.keras")
META_PATH    = Path("model_meta.json")
FRONTEND_PATH = Path("fruit-classifier.html")

for p in [MODEL_PATH, META_PATH, FRONTEND_PATH]:
    if not p.exists():
        raise RuntimeError(
            f"{p} not found. Make sure all files are in the same folder."
        )

# ── Load model + metadata once at startup ─────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(META_PATH) as f:
    meta = json.load(f)

CLASS_NAMES: list = meta["class_names"]
IMG_SIZE: tuple   = tuple(meta["img_size"])
CONFIDENCE_THRESHOLD = 0.60

idx_to_class = {v: k for k, v in meta["class_indices"].items()}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FruitSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Serve the HTML frontend at root ───────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        arr = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    preds = model.predict(arr, verbose=0)[0]

    best_idx   = int(np.argmax(preds))
    prediction = idx_to_class.get(best_idx, CLASS_NAMES[best_idx])
    confidence = float(preds[best_idx])

    probabilities = {
        CLASS_NAMES[i]: round(float(preds[meta["class_indices"][CLASS_NAMES[i]]]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return JSONResponse({
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "probabilities": probabilities,
        "unknown":       confidence < CONFIDENCE_THRESHOLD,
    })


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        str(MODEL_PATH),
        "classes":      CLASS_NAMES,
        "val_accuracy": meta.get("val_accuracy"),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    def open_browser():
        import time; time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    print("\n  FruitSense running -> http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)