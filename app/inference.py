from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, weights_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = Path(weights_path)
        self.model = build_efficientnet_variant().to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.preprocess_image(image).to(self.device)
        logits = self.model(tensor)
        probability = torch.sigmoid(logits).item()
        label = "Detección de OSCC (Cáncer)" if probability > 0.5 else "Tejido epitelial normal"

        return {
            "label": label,
            "confidence": float(probability if probability > 0.5 else 1.0 - probability),
            "probability_oscc": float(probability),
            "probabilities": {
                "Normal": float(1.0 - probability),
                "OSCC": float(probability),
            },
        }

    def score_cam(self, image: Image.Image, top_k: int = 50) -> tuple[np.ndarray, float]:
        img_tensor = self.preprocess_image(image)
        self.model.eval()
        target_layer = self.model.features[-1]

        activations = {}

        def hook_fn(_module, _inp, out):
            activations["a"] = out.detach()

        hook = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = self.model(img_tensor.to(self.device))
            score = torch.sigmoid(logits).item()

        hook.remove()

        acts = activations["a"]
        channels = acts.shape[1]
        height, width = img_tensor.shape[-2:]

        means = acts[0].mean(dim=(1, 2))
        top_indices = torch.topk(means, min(top_k, channels)).indices

        cam = torch.zeros(acts.shape[2:], device=acts.device)

        with torch.no_grad():
            for idx in top_indices:
                mask = acts[0, idx]
                m_min, m_max = mask.min(), mask.max()
                if m_max > m_min:
                    mask = (mask - m_min) / (m_max - m_min)

                mask_upsampled = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

                masked_input = img_tensor.to(self.device) * mask_upsampled
                new_logits = self.model(masked_input)
                new_score = torch.sigmoid(new_logits).item()
                cam += max(0.0, new_score) * acts[0, idx]

        cam = F.relu(cam)
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)

        return cam.cpu().numpy(), float(score)
