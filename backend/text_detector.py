import os
import json
import requests
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Lazy initialization for pipelines
_phishing_pipeline = None
_deepfake_text_pipeline = None

def get_phishing_pipeline():
    global _phishing_pipeline
    if _phishing_pipeline is None:
        logger.info("Loading phishing text detection model...")
        try:
            _phishing_pipeline = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing")
        except Exception as e:
            logger.warning(f"Failed to load primary phishing model, using fallback: {e}")
            _phishing_pipeline = pipeline("text-classification", model="distilbert-base-uncased")
    return _phishing_pipeline

def get_deepfake_text_pipeline():
    global _deepfake_text_pipeline
    if _deepfake_text_pipeline is None:
        logger.info("Loading deepfake text detection model...")
        try:
            _deepfake_text_pipeline = pipeline("text-classification", model="roberta-base-openai-detector")
        except Exception as e:
            logger.warning(f"Failed to load primary text detection model, using fallback: {e}")
            _deepfake_text_pipeline = pipeline("text-classification", model="distilbert-base-uncased")
    return _deepfake_text_pipeline

def detect_phishing(text):
    try:
        pipe = get_phishing_pipeline()
        # Truncate text to max length supported by typical smaller models safely
        result = pipe(text[:512])[0]
        score = result["score"]
        label = result["label"].lower()
        
        # Adjust logic depending on what the model returns
        if "benign" in label or "safe" in label:
            return float(1.0 - score)
        return float(score)
    except Exception as e:
        logger.error(f"Error in detect_phishing: {e}")
        # Fallback to simple heuristic
        if "login" in text.lower() and "password" in text.lower():
            return 0.7
        return 0.1

def detect_deepfake_text(text):
    try:
        pipe = get_deepfake_text_pipeline()
        result = pipe(text[:512])[0]
        score = result["score"]
        label = result["label"].lower()
        
        if "real" in label:
            return float(1.0 - score)
        return float(score)
    except Exception as e:
        logger.error(f"Error in detect_deepfake_text: {e}")
        # Fallback to simple heuristic
        if "urgent" in text.lower() and "money" in text.lower():
            return 0.6
        return 0.3

def detect_text_pro(text):
    """
    Advanced LLM-based PRO detection using OpenRouter.
    """
    llm_res = detect_text_llm(text)
    return {
        "score": llm_res.get("score", 0.5),
        "reason": llm_res.get("reason", "LLM analysis complete.")
    }


def detect_text_llm(text: str):
    """
    Uses OpenRouter API to detect AI-generated or manipulated text.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "score": 0.5,
            "label": "suspicious",
            "reason": "OpenRouter API key missing in environment"
        }

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "Analyze this text for AI generation or manipulation. Return JSON with:\n"
        "{\n"
        "  \"score\": number (0-1),\n"
        "  \"label\": 'human' | 'ai' | 'suspicious',\n"
        "  \"reason\": short explanation\n"
        "}\n\n"
        f"Text to analyze: {text}"
    )
    
    payload = {
        "model": "google/gemma-3-27b-it",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        response.raise_for_status()
        data = response.json()
        
        # OpenRouter returns content in choices[0].message.content
        ai_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # The AI might wrap the JSON in code blocks
        clean_content = ai_message.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
        
        clean_content = clean_content.strip()
            
        try:
            result = json.loads(clean_content)
            return {
                "score": float(result.get("score", 0.5)),
                "label": str(result.get("label", "suspicious")),
                "reason": str(result.get("reason", "LLM Analysis successful"))
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return {
                "score": 0.5,
                "label": "suspicious",
                "reason": "AI response received but JSON parsing failed."
            }
            
    except Exception as e:
        # If API fails → return neutral score 0.5
        return {
            "score": 0.5,
            "label": "neutral",
            "reason": f"OpenRouter API error: {str(e)}"
        }

