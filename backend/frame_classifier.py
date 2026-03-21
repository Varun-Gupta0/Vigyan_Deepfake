import numpy as np
import torch
import timm
from PIL import Image
import cv2
import logging
from torchvision import transforms
import huggingface_hub
import os

logger = logging.getLogger(__name__)


class FrameClassifier:
    def __init__(self, model_type: str = "efficientnet", model_path: str = None):
        logger.info(f"Loading {model_type} model for deepfake detection...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_type = model_type

        if model_type == "xception_ffpp":
            self._load_xception_ffpp(model_path)
        else:
            self._load_efficientnet()

        self.model.eval()
        logger.info(f"{model_type} ready on device: {self.device}")

    def _load_xception_ffpp(self, model_path: str = None):
        import torch
        import timm

        self.model = timm.create_model("xception", pretrained=False, num_classes=1)

        weights_path = model_path or "backend/models/xception_ffpp.pth"

        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print("Loaded Xception FFPP model")
        else:
            logger.warning(f"Weights file not found at {weights_path}")

        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_efficientnet(self):
        repo_id = "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
        try:
            logger.info(f"Downloading/loading JIT model from Hugging Face: {repo_id}")
            model_path = huggingface_hub.hf_hub_download(repo_id, "model.pt")
            self.model = torch.jit.load(model_path, map_location=self.device)
            logger.info("EfficientNet-B7 weights successfully loaded!")
        except Exception as e:
            logger.warning(f"Failed to load JIT weights: {e}")
            logger.info("Falling back to timm EfficientNet-B7 with random weights.")
            self.model = timm.create_model("efficientnet_b7", pretrained=False, num_classes=1)
            self.model = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, frame: np.ndarray) -> float:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        logger.info(
            f"[{self.model_type.upper()}] output={output.item():.4f} prob={float(torch.sigmoid(output).item()):.4f}"
        )
        return float(torch.sigmoid(output).item())
