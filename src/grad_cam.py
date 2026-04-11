import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np


def get_grad_cam(model, input_tensor, target_class, layer_name='layer4'):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    input_tensor.requires_grad = True
    model.zero_grad()

    features = {}
    gradients = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    def grad_hook(name):
        def hook(module, input, output):
            gradients[name] = output
        return hook

    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(hook_fn(name))
            module.register_full_backward_hook(grad_hook(name))

    output = model(input_tensor)

    if output.dim() > 1:
        class_score = output[0, target_class]
    else:
        class_score = output

    class_score.backward()

    for name in features:
        if name in gradients and gradients[name] is not None:
            feature_map = features[name]
            grad = gradients[name][0]
            weights = grad.mean(dim=(1, 2))
            cam = (weights.unsqueeze(-1).unsqueeze(-1) * feature_map).sum(dim=1)
            cam = torch.relu(cam)
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            return cam.detach().cpu().numpy()[0]

    return None


def visualize_grad_cam(image, grad_cam, alpha=0.5):
    grad_cam_resized = cv2.resize(grad_cam, (image.shape[1], image.shape[0]))
    grad_cam_colored = cv2.applyColorMap(np.uint8(255 * grad_cam_resized), cv2.COLORMAP_JET)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, 1 - alpha, grad_cam_colored, alpha, 0)

    return overlay
