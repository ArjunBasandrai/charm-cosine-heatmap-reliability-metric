import base64
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import timm
import gc
import os
import time

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def tensor_to_rgb_uint8(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    m = IMAGENET_MEAN.to(x.device)
    s = IMAGENET_STD.to(x.device)
    x = x * s + m
    x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(0, 2, 3, 1).cpu().detach().numpy()
    return x

def mask_to_heatmap(mask):
    mask = (mask.detach() * 255.0).cpu().numpy().astype(np.uint8)
    outs = []
    for i in range(mask.shape[0]):
        h = cv2.applyColorMap(mask[i], cv2.COLORMAP_JET)
        h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
        outs.append(h)
    return outs

def overlay_heatmap(rgb, heatmap, alpha=0.35):
    out = []
    for i in range(len(rgb)):
        o = cv2.addWeighted(heatmap[i], alpha, rgb[i], 1.0 - alpha, 0)
        out.append(o)
    return out

def _rgb_numpy_to_base64_png(arr_rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf).decode("utf-8")

class EnsembleModel(nn.Module):
    def __init__(self, model_names=None, num_classes=None, pretrained=True):
        super(EnsembleModel, self).__init__()
        if model_names is None:
            model_names = ["efficientnet_b3", "efficientnet_b4", "efficientnet_b5"]
        if num_classes is None:
            num_classes = 5
        self.models = nn.ModuleList([
            self._build_model(name, num_classes, pretrained) for name in model_names
        ])

    def _build_model(self, backbone_name, num_classes, pretrained):
        model = timm.create_model(backbone_name, pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

def get_last_conv_layer(m):
    last = None
    for module in m.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last

def get_submodel_and_layer(m, submodel_index=0):
    sub = m.models[submodel_index] if isinstance(m, EnsembleModel) else m
    if hasattr(sub, "act2"):                      # efficientnet in timm
        return sub, sub.act2
    if hasattr(sub, "blocks") and len(sub.blocks) > 0:
        last = sub.blocks[-1]
        for name in ("act3", "act2", "conv_pwl", "conv_dw"):
            if hasattr(last, name):
                return sub, getattr(last, name)
    return sub, get_last_conv_layer(sub)

class GradCAMOnce:
    def __init__(self, submodel, layer):
        self.sub = submodel
        self.layer = layer
        self.activations = None
        self.gradients = None
        self.fh = self.layer.register_forward_hook(self._fhook)
        self.bh = self.layer.register_full_backward_hook(self._bhook)

    def _fhook(self, module, inp, out):
        self.activations = out

    def _bhook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.fh.remove()
        self.bh.remove()
        self.activations = None
        self.gradients = None

    def cam(self, inputs, class_idx=None):
        self.sub.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.sub(inputs)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            scores = logits.gather(1, class_idx.view(-1, 1)).sum()
            scores.backward()

        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        flat = cam.flatten(1)
        cmin = flat.min(dim=1)[0].view(-1,1,1,1)
        cmax = flat.max(dim=1)[0].view(-1,1,1,1)
        cam = (cam - cmin) / (cmax - cmin + 1e-6)

        self.activations = None
        self.gradients = None
        self.sub.zero_grad(set_to_none=True)

        del logits, scores, acts, grads, weights, flat, cmin, cmax
        return cam.squeeze(1), class_idx

def charm(images, device, model, win=15, alpha=0.9):
    eps = 1e-8
    invert_T = False
    k = torch.ones(1, 1, win, win, device=device) / (win * win)

    def norm01(t, dims=(-2, -1), keepdim=True, eps=1e-8):
        tmin = t.amin(dim=dims, keepdim=keepdim)
        tmax = t.amax(dim=dims, keepdim=keepdim)
        return (t - tmin) / (tmax - tmin + eps)

    images = images.to(device)

    g = images[:, 1:2, ...]
    g = (g - g.amin(dim=(-2, -1), keepdim=True))
    g = g / (g.amax(dim=(-2, -1), keepdim=True) - g.amin(dim=(-2, -1), keepdim=True) + eps)
    g = torch.clamp(g, 0.0, 1.0)

    mu   = F.conv2d(g, k, padding=win//2)
    g2   = g * g
    mu2  = F.conv2d(g2, k, padding=win//2)
    var  = torch.clamp(mu2 - mu * mu, min=0.0)
    sigma = torch.sqrt(var + eps)
    mad   = F.conv2d((g - mu).abs(), k, padding=win//2)

    T_soft = norm01(mu)
    if invert_T:
        T_soft = 1.0 - T_soft
    I_soft = norm01(mad)
    F_soft = 1 - T_soft

    threshold = 0.5
    T_map = (T_soft * (T_soft > threshold)).float().squeeze(1)
    I_map = (I_soft * (I_soft > threshold)).float().squeeze(1)
    F_map = (F_soft * (F_soft > threshold)).float().squeeze(1)

    sub, target_layer = get_submodel_and_layer(model)
    gcam = GradCAMOnce(sub.to(device), target_layer)
    with torch.no_grad():
        preds = sub(images).argmax(dim=1)
    gmap, _ = gcam.cam(images, class_idx=preds)
    gcam.close()

    idx = 0
    rgb = tensor_to_rgb_uint8(images[idx])[0]
    T_heat = mask_to_heatmap(T_map.squeeze(1)[idx:idx+1])[0]
    I_heat = mask_to_heatmap(I_map.squeeze(1)[idx:idx+1])[0]
    F_heat = mask_to_heatmap(F_map.squeeze(1)[idx:idx+1])[0]
    G_heat = mask_to_heatmap(gmap[idx:idx+1])[0]

    T_over = overlay_heatmap([rgb], [T_heat], alpha=alpha)[0]
    I_over = overlay_heatmap([rgb], [I_heat], alpha=alpha)[0]
    F_over = overlay_heatmap([rgb], [F_heat], alpha=alpha)[0]
    G_over = overlay_heatmap([rgb], [G_heat], alpha=alpha)[0]

    def cos(a,b): return (a*b).sum() / (a.norm(p=2)*b.norm(p=2) + eps)
    G_b = gmap[idx]
    T_b = T_soft.squeeze(1)[idx]
    I_b = I_soft.squeeze(1)[idx]
    F_b = F_soft.squeeze(1)[idx]
    wT, wF, wI = 1.0, 0.3, 0.2
    k_ = 3.0
    z = k_ * (wT*cos(G_b, T_b) - wF*cos(G_b, F_b) - wI*cos(G_b, I_b))
    R = torch.sigmoid(z).clamp(0,1)

    del mu, g2, mu2, var, sigma, mad, T_soft, I_soft, F_soft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "T": _rgb_numpy_to_base64_png(T_over),
        "I": _rgb_numpy_to_base64_png(I_over),
        "F": _rgb_numpy_to_base64_png(F_over),
        "G": _rgb_numpy_to_base64_png(G_over),
        "R": round(float(R.item()), 4),
    }
