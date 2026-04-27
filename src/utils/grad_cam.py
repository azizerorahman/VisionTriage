import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path


# ── Grad-CAM ───────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer, model_type='monocular'):
        self.model        = model
        self.target_layer = target_layer
        self.model_type   = model_type

        # Storage for hooks
        self._activations = {}
        self._gradients   = {}

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations['layer4'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients['layer4'] = grad_output[0].detach()

        self._forward_handle  = self.target_layer.register_forward_hook(
            forward_hook
        )
        self._backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_left, class_idx=None,
                 input_right=None, stream='left'):
        self.model.eval()
        device = next(self.model.parameters()).device

        input_left = input_left.to(device)
        input_left.requires_grad_(True)

        # Clear stored values
        self._activations = {}
        self._gradients   = {}

        self.model.zero_grad()

        if self.model_type == 'monocular':
            output = self.model(input_left)

        elif self.model_type in ('late_concat', 'cross_attention'):
            if input_right is None:
                # Generate synthetic right eye if not provided
                input_right = torch.flip(input_left, dims=[3])
            input_right = input_right.to(device)
            output = self.model(input_left, input_right)

        else:
            raise ValueError(
                f"Unknown model_type: '{self.model_type}'. "
                f"Use 'monocular', 'late_concat', or 'cross_attention'"
            )

        # Get predicted class and probabilities
        probs      = torch.softmax(output, dim=1)
        pred_class = int(output.argmax(dim=1).item())
        pred_probs = probs.detach().cpu().numpy()[0]

        if class_idx is None:
            class_idx = pred_class

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # ── Compute Grad-CAM ──────────────────────────────────────────
        if 'layer4' not in self._activations:
            raise RuntimeError(
                "No activations captured."
            )
        if 'layer4' not in self._gradients:
            raise RuntimeError(
                "No gradients captured."
            )

        activations = self._activations['layer4']
        gradients = self._gradients['layer4']
        alpha = gradients.mean(dim=[2, 3])
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # ── Normalise ───────────────────────────────────────
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # ── Upsample ──────────────────────────────
        heatmap = cv2.resize(
            cam,
            (480, 480),
            interpolation=cv2.INTER_LINEAR
        )

        return heatmap, pred_class, pred_probs

    def generate_overlay(self, image_rgb, heatmap, alpha=0.4):
        heatmap_uint8  = np.uint8(255 * heatmap)
        heatmap_colour = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb    = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlay = cv2.addWeighted(
            image_rgb.astype(np.uint8), 1 - alpha,
            heatmap_rgb.astype(np.uint8), alpha,
            0
        )

        return overlay


# ── IoU Computation ───────────────────────────────────────────────────

def compute_iou(heatmap, lesion_mask, threshold=0.5):
    binary_cam  = (heatmap >= threshold).astype(np.uint8)
    binary_mask = (lesion_mask > 0).astype(np.uint8)

    # Resize mask to match heatmap if needed
    if binary_mask.shape != binary_cam.shape:
        binary_mask = cv2.resize(
            binary_mask,
            (binary_cam.shape[1], binary_cam.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    intersection = np.logical_and(binary_cam, binary_mask).sum()
    union        = np.logical_or(binary_cam, binary_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_iou_batch(heatmaps, lesion_masks, threshold=0.5):
    assert len(heatmaps) == len(lesion_masks), (
        "heatmaps and lesion_masks must have same length"
    )

    iou_scores = [
        compute_iou(h, m, threshold)
        for h, m in zip(heatmaps, lesion_masks)
    ]

    return float(np.mean(iou_scores)), iou_scores

def compute_spatial_correlation(cam_a, cam_b):
    flat_a = cam_a.flatten().astype(np.float64)
    flat_b = cam_b.flatten().astype(np.float64)

    # Handle zero-variance case
    if flat_a.std() < 1e-8 or flat_b.std() < 1e-8:
        return 0.0

    r = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return r

def get_gradcam_for_model(model, model_type):
    if model_type == 'monocular':
        # M1
        target_layer = model.backbone[-1]

    elif model_type == 'late_concat':
        # M2
        target_layer = model.stream_L[-1]

    elif model_type == 'cross_attention':
        # M3
        target_layer = model.left_encoder[-1]

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Use 'monocular', 'late_concat', or 'cross_attention'"
        )

    return GradCAM(
        model=model,
        target_layer=target_layer,
        model_type=model_type,
    )

def is_implausible_activation(heatmap, image_rgb,
                              border_fraction=0.1,
                              activation_threshold=0.5):
    H, W = heatmap.shape

    # Create border mask
    border_pixels = int(min(H, W) * border_fraction)
    border_mask   = np.zeros((H, W), dtype=bool)
    border_mask[:border_pixels, :]   = True   # Top
    border_mask[-border_pixels:, :]  = True   # Bottom
    border_mask[:, :border_pixels]   = True   # Left
    border_mask[:, -border_pixels:]  = True   # Right

    # High activation pixels
    high_activation = heatmap >= activation_threshold

    total_high    = high_activation.sum()
    border_high   = (high_activation & border_mask).sum()

    if total_high == 0:
        return True, 1.0   # No activation at all = implausible

    border_ratio = float(border_high / total_high)
    
    is_implausible = border_ratio > 0.6

    return is_implausible, border_ratio