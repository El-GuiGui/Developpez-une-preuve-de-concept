from io import BytesIO
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="Cityscapes Segmentation API", version="1.0")

SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.keras")).resolve()
MODEL_URL = os.getenv("MODEL_URL", "").strip()


def _download_model_if_needed():
    if MODEL_PATH.exists():
        return
    if not MODEL_URL:
        return

    import requests

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    r = requests.get(MODEL_URL, stream=True, timeout=300, headers=headers)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


if not SKIP_MODEL_LOAD:
    from scripts.inference import predict_from_pil, ensure_model_present
else:
    import numpy as np

    def predict_from_pil(img: Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)

        mask_color = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")

        overlay_img = img.copy()

        info = {"dummy": True, "shape": [h, w]}
        return mask, mask_color, overlay_img, info


@app.on_event("startup")
def _startup():
    if SKIP_MODEL_LOAD:
        return
    ensure_model_present(MODEL_PATH, force=False)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "skip_model_load": SKIP_MODEL_LOAD,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)):
    """
    Return: PNG grayscale mask (values 0..7)
    """
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw))
        mask, _, _, _ = predict_from_pil(img, model_path=MODEL_PATH)

        out = Image.fromarray(mask, mode="L")
        buf = BytesIO()
        out.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/mask_color")
async def predict_mask_color(file: UploadFile = File(...)):
    """
    Return: PNG RGB colorized mask
    """
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw))
        _, mask_color, _, _ = predict_from_pil(img, model_path=MODEL_PATH)

        buf = BytesIO()
        mask_color.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/overlay")
async def predict_overlay(file: UploadFile = File(...)):
    """
    Return: PNG overlay (RGB + mask color)
    """
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw))
        _, _, overlay_img, _ = predict_from_pil(img, model_path=MODEL_PATH)

        buf = BytesIO()
        overlay_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
