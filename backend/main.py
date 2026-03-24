# pyre-ignore-all-errors
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.video_detector import VideoDeepfakeDetector
from backend.frame_classifier import FrameClassifier
from backend.text_detector import detect_phishing, detect_deepfake_text, detect_text_pro
from backend.fusion_engine import fusion_engine, calibrate
from backend.decision_engine import decision_engine
from backend.explainability import explainability
import os
import tempfile
import logging
import time
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VERI-AI EDGE",
    description="Autonomous Multimodal Deepfake Detection API",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
video_detector = VideoDeepfakeDetector()
current_model_name = "Lite-Heuristic"  # Dynamic – updated on every request


def _update_model_name(mode: str) -> str:
    """Update the global current_model_name based on mode and return it."""
    global current_model_name
    if mode == "pro":
        current_model_name = "Xception-FFPP"
    else:
        current_model_name = "Lite-Heuristic"
    return current_model_name


@app.on_event("startup")
async def startup_event():
    logger.info("VERI-AI EDGE startup complete.")

@app.get("/")
async def root():
    return JSONResponse(content={
        "message": "Welcome to the VERI-AI EDGE API",
        "docs": "/docs",
        "health": "/health",
        "status": "online"
    })

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={})


# ── helpers ───────────────────────────────────────────────────────────────────

def _verdict(label: str) -> str:
    return {"FAKE": "deepfake", "REAL": "authentic"}.get(label, "suspicious")


def _label_and_conf(fake_prob: float):
    """Return (api_label, confidence_pct) from a fake_probability value."""
    real_prob = 1.0 - fake_prob
    if fake_prob > 0.70:
        return "FAKE", round(fake_prob * 100, 1)
    elif fake_prob < 0.30:
        return "REAL", round(real_prob * 100, 1)
    else:
        return "UNCERTAIN", round(max(fake_prob, real_prob) * 100, 1)


# ── /analyze/video ────────────────────────────────────────────────────────────

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), mode: str = Form("lite")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        model_name = _update_model_name(mode)
        start = time.time()

        result = video_detector.detect(video_path, mode=mode)
        analysis_time = round(time.time() - start, 2)

        try:
            os.remove(video_path)
        except OSError:
            pass

        fake_prob       = float(result.get("score", 0.5))
        frames_analyzed = int(result.get("framesAnalyzed", 0))
        faces           = result.get("faces", [])

        fused_score     = fusion_engine(fake_prob, 0.0, face_count=len(faces))
        calibrated_prob = calibrate(fused_score)

        api_label, confidence_pct = _label_and_conf(calibrated_prob)
        decision    = decision_engine(calibrated_prob)
        explanation = explainability(calibrated_prob, data_type="video", data=video_path, mode=mode)

        return JSONResponse(content={
            "label":                   api_label,
            "confidence":              confidence_pct,
            "verdict":                 _verdict(api_label),
            "fake_probability":        round(calibrated_prob, 4),
            "real_probability":        round(1.0 - calibrated_prob, 4),
            "mean_fake_probability":   round(fake_prob, 4),
            "median_fake_probability": round(fake_prob, 4),
            "std_fake_probability":    0.0,
            "frame_scores":            [],
            "decision":                decision.to_dict(),
            "reason":                  explanation.get("reason", "Video analysis complete"),
            "faces":                   faces,
            "processing_steps":        explanation.get("processing_steps", []),
            "metrics": {
                "framesAnalyzed": frames_analyzed,
                "facesDetected":  len(faces),
                "latency":        round(analysis_time * 1000, 2),
                "modelMode":      mode,
                "inferenceType":  "cloud" if mode == "pro" else "local",
                "fps":            round(frames_analyzed / analysis_time, 2) if analysis_time > 0 else 0.0,
                "modelUsed":      model_name,
                "model":          model_name,
            },
        })

    except Exception as e:
        logger.error(f"/analyze/video error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── /analyze/frame ────────────────────────────────────────────────────────────

@app.post("/analyze/frame")
async def analyze_frame(request: Request):
    try:
        start_time = time.time()
        body       = await request.json()
        image_b64  = body.get("image", "")
        if not image_b64:
            return JSONResponse(content={"error": "No image provided"}, status_code=400)

        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(image_b64), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        mode       = body.get("mode", "lite")
        model_name = _update_model_name(mode)
        result     = video_detector.classify_frame_live(frame, mode=mode)
        faces      = result.get("faces", [])
        fake_prob  = float(result.get("score", 0.0))

        if not result.get("face_found"):
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return JSONResponse(content={
                "label":                   "REAL",
                "confidence":              100.0,
                "verdict":                 "authentic",
                "fake_probability":        0.0,
                "real_probability":        1.0,
                "mean_fake_probability":   0.0,
                "median_fake_probability": 0.0,
                "std_fake_probability":    0.0,
                "frame_scores":            [],
                "decision":                {"label": "REAL", "confidence": 100.0},
                "faces":                   [],
                "metrics": {
                    "framesAnalyzed": 1,
                    "facesDetected":  0,
                    "latency":        latency_ms,
                    "modelMode":      mode,
                    "inferenceType":  "cloud" if mode == "pro" else "local",
                    "fps":            0.0,
                    "modelUsed":      model_name,
                    "model":          model_name,
                },
            })

        fused_score     = fusion_engine(fake_prob, 0.0, face_count=len(faces))
        calibrated_prob = calibrate(fused_score)

        api_label, confidence_pct = _label_and_conf(calibrated_prob)
        decision    = decision_engine(calibrated_prob)
        explanation = explainability(calibrated_prob, data_type="video", data=None, mode=mode)

        analysis_time = time.time() - start_time
        latency_ms    = round(analysis_time * 1000, 2)

        return JSONResponse(content={
            "label":                   api_label,
            "confidence":              confidence_pct,
            "verdict":                 _verdict(api_label),
            "fake_probability":        round(calibrated_prob, 4),
            "real_probability":        round(1.0 - calibrated_prob, 4),
            "mean_fake_probability":   round(calibrated_prob, 4),
            "median_fake_probability": round(calibrated_prob, 4),
            "std_fake_probability":    0.0,
            "frame_scores":            [round(calibrated_prob, 4)],
            "decision":                decision.to_dict(),
            "reason":                  explanation.get("reason", "Live analysis complete"),
            "faces":                   faces,
            "metrics": {
                "framesAnalyzed": 1,
                "facesDetected":  len(faces),
                "latency":        latency_ms,
                "modelMode":      mode,
                "inferenceType":  "cloud" if mode == "pro" else "local",
                "fps":            round(1.0 / analysis_time, 2) if analysis_time > 0 else 0.0,
                "modelUsed":      model_name,
                "model":          model_name,
            },
        })

    except Exception as e:
        logger.error(f"/analyze/frame error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── /analyze/text ─────────────────────────────────────────────────────────────

@app.post("/analyze/text")
async def analyze_text(request: Request, file: UploadFile = File(None)):
    try:
        start_time   = time.time()
        content_type = request.headers.get("content-type", "")
        mode = "lite"
        if file is not None:
            text = (await file.read()).decode("utf-8")
        elif "application/json" in content_type:
            body = await request.json()
            text = body.get("text", "")
            mode = body.get("mode", "lite")
        else:
            text = (await request.body()).decode("utf-8")

        phishing_score = detect_phishing(text)

        # Determine text-specific model name
        if mode == "pro":
            text_model_name = "Gemma-3-27B"
            res = detect_text_pro(text)
            deepfake_score = res["score"]
            llm_reason     = res["reason"]
        else:
            text_model_name = "RoBERTa/DistilRoBERTa"
            deepfake_score = detect_deepfake_text(text)
            llm_reason = None

        # Also update global model name for /health
        _update_model_name(mode)

        fused_score     = fusion_engine(0.0, deepfake_score)
        calibrated_prob = calibrate(fused_score)
        decision        = decision_engine(calibrated_prob)
        explanation     = explainability(calibrated_prob, data_type="text", data=text, mode=mode)
        api_label, confidence_pct = _label_and_conf(calibrated_prob)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return JSONResponse(content={
            "label":            api_label,
            "confidence":       confidence_pct,
            "verdict":          _verdict(api_label),
            "fake_probability": round(calibrated_prob, 4),
            "real_probability": round(1.0 - calibrated_prob, 4),
            "decision":         decision.to_dict(),
            "reason":           llm_reason if (mode == "pro" and llm_reason) else explanation.get("reason", "Text analysis complete"),
            "phishing_score":   round(phishing_score * 100, 1),
            "processing_steps": explanation.get("processing_steps", []),
            "metrics": {
                "framesAnalyzed": 0,
                "facesDetected":  0,
                "latency":        latency_ms,
                "modelMode":      mode,
                "inferenceType":  "cloud" if mode == "pro" else "local",
                "fps":            0.0,
                "modelUsed":      text_model_name,
                "model":          text_model_name,
            },
        })

    except Exception as e:
        logger.error(f"/analyze/text error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse(content={
        "status":            "ok",
        "edgeMode":          "enabled",
        "inferenceDevice":   "CPU/MPS/CUDA",
        "latency":           0,
        "model":             current_model_name,
        "liveStreamEnabled": True,
        "metrics": {
            "framesAnalyzed": 0,
            "facesDetected":  0,
            "latency":        0.0,
            "modelMode":      "pro" if current_model_name == "Xception-FFPP" else "lite",
            "inferenceType":  "local",
            "fps":            0.0,
            "modelUsed":      current_model_name,
            "model":          current_model_name,
        },
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
