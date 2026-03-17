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


def explainability(fusion_result: float, data_type: str = 'video', data: Any = None) -> Dict[str, Any]:
    if data_type == 'video':
        explainer = VideoExplainer()
        return explainer.generate_explanation(np.array([]), fusion_result)
    elif data_type == 'text':
        explainer = TextExplainer()
        return explainer.generate_explanation(data or "", fusion_result)
    else:
        return {
            "explanation": f"Unsupported data type: {data_type}",
            "reason": f"Unsupported data type: {data_type}",
            "processing_steps": [],
        }
