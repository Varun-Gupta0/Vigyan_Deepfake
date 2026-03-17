import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

# Load the model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Define the phishing detection function
def detect_phishing(text):
    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the probability of the text being phishing
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    phishing_probability = probabilities[0][1].item()  # Assuming the second class is phishing
    
    # Determine the reason based on the probability
    if phishing_probability > 0.5:
        reason = "Suspicious language pattern"
    else:
        reason = "No significant phishing patterns detected"
    
    return {
        "text_score": phishing_probability,
        "reason": reason
    }

# Placeholder deepfake detection logic
def detect_deepfake_text(text):
    score = len(re.findall(r'\bdeepfake\b', text, re.IGNORECASE))
    return score
