from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_efficientnet_variant() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model


class HistopathologyPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_efficientnet_variant().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self._process_image(image)
        logits = self.model(tensor)
        probability = float(torch.sigmoid(logits).item())
        label = "Detección de OSCC (Cáncer)" if probability > 0.5 else "Tejido epitelial normal"
        confidence = probability if probability > 0.5 else 1.0 - probability

        return {
            "label": label,
            "confidence": float(confidence),
            "probability_oscc": probability,
            "probabilities": {
                "Normal": float(1.0 - probability),
                "OSCC": probability,
            },
        }


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model_efficientnet_variante.pth"
PREDICTOR = HistopathologyPredictor(MODEL_PATH)


def predict_histopathology(image: Image.Image) -> dict:
    return PREDICTOR.predict(image)
