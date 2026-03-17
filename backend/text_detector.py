def detect_phishing(text):
    # Placeholder for phishing detection logic
    if "login" in text.lower() and "password" in text.lower():
        return 0.7
    else:
        return 0.1

def detect_deepfake_text(text):
    # Placeholder for deepfake text detection logic
    if "urgent" in text.lower() and "money" in text.lower():
        return 0.6
    else:
        return 0.2
