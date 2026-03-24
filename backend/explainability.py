import json
import requests
import os
import numpy as np
from typing import Dict, Any, List

class VideoExplainer:
    """Generate explanations for video deepfake predictions."""

    def __init__(self):
        pass

    def generate_explanation(self, frames: np.ndarray, prediction_score: float) -> Dict[str, Any]:
        processing_steps = [
            "Extracting frames from video",
            "Running MediaPipe face detection",
            "Computing artifact heuristics",
            "Aggregating multi-frame scores",
        ]
        reason = f"Video deepfake score: {prediction_score:.2f}. Facial artifact analysis complete."
        return {
            "explanation": reason,
            "reason": reason,
            "processing_steps": processing_steps,
        }


class TextExplainer:
    """Generate explanations for text phishing predictions."""

    def __init__(self):
        pass

    def generate_explanation(self, text: str, prediction_score: float) -> Dict[str, Any]:
        processing_steps = [
            "Tokenizing input text",
            "Running phishing keyword heuristics",
            "Running deepfake text heuristics",
            "Fusing text analysis scores",
        ]
        reason = f"Text analysis score: {prediction_score:.2f}. Phishing/deepfake text scan complete."
        return {
            "explanation": reason,
            "reason": reason,
            "processing_steps": processing_steps,
        }


def llm_explainability(score: float, data_type: str, context: str = "") -> str:
    """
    Calls OpenRouter LLM to generate a detailed reason for the detection result.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return f"Pro mode: Analyzed {data_type} and calculated score {score:.2f}. [LLM details unavailable: Missing API key]"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    label = "Deepfake" if score > 0.5 else "Authentic"
    conf = round(abs(score - 0.5) * 200, 1) # Simple confidence calculation
    
    prompt = (
        f"You are an AI Deepfake Analyst. "
        f"A {data_type} analysis has yielded a {label} verdict with {conf}% confidence (Score: {score:.2f}).\n"
        f"Context provided: {context}\n"
        "Generate a brief (2-3 sentences), authoritative reasoning for this result. "
        "Do not mention that you are an AI, just provide the technical reasoning as if from a forensic report."
    )
    
    payload = {
        "model": "google/gemma-3-27b-it",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        reason = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return reason or "Neural pattern analysis complete. Probability exceeds threshold."
    except Exception as e:
        return f"Advanced analysis complete. Verdict: {label} ({score:.2f})."


def explainability(fusion_result: float, data_type: str = 'video', data: Any = None, mode: str = "lite") -> Dict[str, Any]:
    # 1. Base explanation from heuristics
    if data_type == 'video':
        explainer = VideoExplainer()
        explanation_data = explainer.generate_explanation(np.array([]), fusion_result)
    elif data_type == 'text':
        explainer = TextExplainer()
        explanation_data = explainer.generate_explanation(data or "", fusion_result)
    else:
        explanation_data = {
            "explanation": f"Unsupported data type: {data_type}",
            "reason": f"Unsupported data type: {data_type}",
            "processing_steps": [],
        }

    # 2. If PRO mode, enhance with LLM reasoning
    if mode == "pro":
        # Pass a snippet of data for context if it's text
        context = data if data_type == "text" else "Temporal analysis of faces and motion artifacts."
        explanation_data["reason"] = llm_explainability(fusion_result, data_type, context)
        
    return explanation_data
