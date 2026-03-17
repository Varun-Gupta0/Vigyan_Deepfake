from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.video_detector import VideoDeepfakeDetector
from backend.text_detector import detect_phishing, detect_deepfake_text
from backend.fusion_engine import fusion_engine
from backend.decision_engine import decision_engine
from backend.explainability import explainability
import os
import tempfile
import logging
import time
import base64
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VERI-AI EDGE",
    description="Autonomous Multimodal Deepfake Detection API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_detector = VideoDeepfakeDetector()


@app.on_event("startup")
async def startup_event():
    logger.info("VERI-AI EDGE startup complete.")


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
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video_path = tmp.name

        start = time.time()
        result = video_detector.detect(video_path)
        analysis_time = round(time.time() - start, 2)

        try:
            os.remove(video_path)
        except OSError:
            pass

        fake_prob  = float(result["video_score"])
        mean_prob  = float(result.get("mean_fake_probability", fake_prob))
        median_prob = float(result.get("median_fake_probability", fake_prob))
        std_prob   = float(result.get("std_fake_probability", 0.0))
        frame_scores = [round(s, 4) for s in result.get("frame_scores", [])]
        frames_analyzed = int(result.get("frames_analyzed", 0))
        faces      = result.get("faces", [])
        reason     = result.get("reason", "Temporal analysis complete")

        api_label, confidence_pct = _label_and_conf(fake_prob)
        decision = decision_engine(fake_prob)
        explanation = explainability(fake_prob, data_type="video", data=video_path)

        return JSONResponse(content={
            "label":                  api_label,
            "confidence":             confidence_pct,
            "verdict":                _verdict(api_label),
            "fake_probability":       round(fake_prob, 4),
            "real_probability":       round(1.0 - fake_prob, 4),
            "mean_fake_probability":  round(mean_prob, 4),
            "median_fake_probability": round(median_prob, 4),
            "std_fake_probability":   round(std_prob, 4),
            "frame_scores":           frame_scores,
            "decision":               decision.to_dict(),
            "reason":                 reason,
            "faces":                  faces,
            "analysis_time_seconds":  analysis_time,
            "processing_steps":       explanation.get("processing_steps", []),
            "metrics": {
                "framesAnalyzed":  frames_analyzed,
                "processingTime":  analysis_time,
                "modelUsed":       "EfficientNet-B7 + MediaPipe (Temporal)",
                "inferenceDevice": "CPU/MPS/CUDA",
            },
        })

    except Exception as e:
        logger.error(f"/analyze/video error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── /analyze/frame ────────────────────────────────────────────────────────────

@app.post("/analyze/frame")
async def analyze_frame(request: Request):
    try:
        body = await request.json()
        image_b64 = body.get("image", "")
        if not image_b64:
            return JSONResponse(content={"error": "No image provided"}, status_code=400)

        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(image_b64), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        result = video_detector.classify_frame_live(frame)
        faces = result.get("faces", [])
        fake_prob = float(result.get("score", 0.0))

        if not result.get("face_found"):
            # No face → cannot make a determination → treat as real
            return JSONResponse(content={
                "label":                  "REAL",
                "confidence":             100.0,
                "verdict":                "authentic",
                "fake_probability":       0.0,
                "real_probability":       1.0,
                "mean_fake_probability":  0.0,
                "median_fake_probability": 0.0,
                "std_fake_probability":   0.0,
                "frame_scores":           [],
                "decision":               {"label": "REAL", "confidence": 100.0},
                "faces":                  [],
                "metrics": {
                    "facesDetected": 0,
                    "modelUsed":     "EfficientNet-B7 + MediaPipe",
                    "inferenceDevice": "CPU/MPS/CUDA",
                },
            })

        api_label, confidence_pct = _label_and_conf(fake_prob)
        decision = decision_engine(fake_prob)

        return JSONResponse(content={
            "label":                  api_label,
            "confidence":             confidence_pct,
            "verdict":                _verdict(api_label),
            "fake_probability":       round(fake_prob, 4),
            "real_probability":       round(1.0 - fake_prob, 4),
            "mean_fake_probability":  round(fake_prob, 4),
            "median_fake_probability": round(fake_prob, 4),
            "std_fake_probability":   0.0,
            "frame_scores":           [round(fake_prob, 4)],
            "decision":               decision.to_dict(),
            "faces":                  faces,
            "metrics": {
                "facesDetected": len(faces),
                "modelUsed":     "EfficientNet-B7 + MediaPipe",
                "inferenceDevice": "CPU/MPS/CUDA",
            },
        })

    except Exception as e:
        logger.error(f"/analyze/frame error: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── /analyze/text ─────────────────────────────────────────────────────────────

@app.post("/analyze/text")
async def analyze_text(request: Request, file: UploadFile = File(None)):
    try:
        content_type = request.headers.get("content-type", "")
        if file is not None:
            text = (await file.read()).decode("utf-8")
        elif "application/json" in content_type:
            body = await request.json()
            text = body.get("text", "")
        else:
            text = (await request.body()).decode("utf-8")

        phishing_score  = detect_phishing(text)
        deepfake_score  = detect_deepfake_text(text)
        fused_score     = fusion_engine(0.0, deepfake_score)
        fake_prob       = float(fused_score)
        decision        = decision_engine(fake_prob)
        explanation     = explainability(fake_prob, data_type="text", data=text)
        api_label, confidence_pct = _label_and_conf(fake_prob)

        return JSONResponse(content={
            "label":            api_label,
            "confidence":       confidence_pct,
            "verdict":          _verdict(api_label),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1.0 - fake_prob, 4),
            "decision":         decision.to_dict(),
            "reason":           explanation.get("reason", "Text analysis complete"),
            "phishing_score":   round(phishing_score * 100, 1),
            "processing_steps": explanation.get("processing_steps", []),
            "metrics": {
                "framesAnalyzed":  0,
                "processingTime":  0.0,
                "modelUsed":       "Keyword Heuristic",
                "inferenceDevice": "CPU",
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
        "model":             "EfficientNet-B7",
        "liveStreamEnabled": True,
    })
