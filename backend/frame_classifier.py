import numpy as np
import torch
import timm
from PIL import Image
import cv2
import logging
from torchvision import transforms
import huggingface_hub

logger = logging.getLogger(__name__)


class FrameClassifier:
    def __init__(self):
        logger.info("Loading EfficientNet-B7 model for deepfake detection...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        repo_id = "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
        try:
            logger.info(f"Downloading/loading JIT model from Hugging Face: {repo_id}")
            model_path = huggingface_hub.hf_hub_download(repo_id, "model.pt")
            self.model = torch.jit.load(model_path, map_location=self.device)
            logger.info("Deepfake weights successfully loaded!")
        except Exception as e:
            logger.warning(f"Failed to load JIT weights: {e}")
            logger.info("Falling back to timm EfficientNet-B7 with random weights.")
            self.model = timm.create_model("efficientnet_b7", pretrained=False, num_classes=1)
            self.model = self.model.to(self.device)

        self.model.eval()

        # Preprocessing: BGR→RGB is handled in classify().
        # ImageNet normalization matches the facetorch training pipeline.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"EfficientNet-B7 ready on device: {self.device}")

    def classify(self, frame: np.ndarray) -> float:
        """
        Accept a BGR numpy frame (from OpenCV), return fake_probability in [0, 1].

        The facetorch model was trained so that:
          - high sigmoid output → REAL face
          - low sigmoid output  → FAKE face
        Therefore: fake_probability = 1 - sigmoid(logit)
        """
        # Convert BGR (OpenCV) → RGB before normalising
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(input_tensor)
            p_real = torch.sigmoid(logit).item()
            fake_prob = 1.0 - p_real

        logger.info(
            f"[INFERENCE] logit={logit.item():.4f}  "
            f"p_real={p_real:.4f}  fake_prob={fake_prob:.4f}"
        )
        return fake_prob
