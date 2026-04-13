import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_efficientnet_variant():
    """Define la arquitectura exacta utilizada en el entrenamiento."""
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    return model

def load_model(weights_path, device):
    """Carga los pesos en la arquitectura definida."""
    model = get_efficientnet_variant()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    """Preprocesa la imagen PIL para el modelo."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return transform(image).unsqueeze(0)

def denormalize(t: torch.Tensor) -> np.ndarray:
    """Convierte un tensor normalizado a una imagen numpy visible."""
    img = t.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * STD + MEAN, 0, 1).astype(np.float32)

def score_cam(model, img_tensor: torch.Tensor, device, top_k: int = 50) -> np.ndarray:
    """Implementación de Score-CAM para interpretabilidad."""
    model.eval()
    target_layer = model.features[-1]
    
    acts_dict = {}
    def hook_fn(m, i, o):
        acts_dict['a'] = o.detach()
        
    hook = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        score = torch.sigmoid(logits).item()
    
    hook.remove()
    
    acts = acts_dict['a']  # [1, C, h, w]
    C = acts.shape[1]
    H, W = img_tensor.shape[-2:]
    
    # Seleccionar top canales por magnitud de activación para eficiencia
    act_means = acts[0].mean(dim=(1, 2))
    top_indices = torch.topk(act_means, min(top_k, C)).indices
    
    cam = torch.zeros(acts.shape[2:], device=acts.device)
    
    with torch.no_grad():
        for i in top_indices:
            # Mapas de activación normalizados
            mask = acts[0, i]
            m_min, m_max = mask.min(), mask.max()
            if m_max > m_min:
                mask = (mask - m_min) / (m_max - m_min)
            
            # Upsample mask to input size
            mask_upsampled = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # Mask input and get new score
            masked_input = img_tensor.to(device) * mask_upsampled
            new_logits = model(masked_input)
            new_score = torch.sigmoid(new_logits).item()
            
            # Weight is the score difference (CIC - Class Interaction Capability)
            weight = max(0, new_score) 
            cam += weight * acts[0, i]
            
    cam = F.relu(cam)
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        cam = (cam - c_min) / (c_max - c_min)
        
    return cam.cpu().numpy(), score
