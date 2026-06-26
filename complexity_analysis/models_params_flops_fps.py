import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from thop import profile
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation
)

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {DEVICE}")

# =========================================================
# DENSENET
# =========================================================
class LateFusionDenseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.rgb_net = smp.Unet(
            encoder_name="densenet121",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        self.ms_net = smp.Unet(
            encoder_name="densenet121",
            encoder_weights=None,
            in_channels=6,
            classes=2
        )

    def forward(self, rgb, ms):

        return self.rgb_net(rgb) + self.ms_net(ms)

# =========================================================
# UNET++
# =========================================================
class LateFusionUNetPP(nn.Module):

    def __init__(self):

        super().__init__()

        self.rgb_net = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        self.ms_net = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=6,
            classes=2
        )

    def forward(self, rgb, ms):

        return self.rgb_net(rgb) + self.ms_net(ms)

# =========================================================
# PSPNET
# =========================================================
class LateFusionPSPNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.rgb_net = smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        self.ms_net = smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=6,
            classes=2
        )

    def forward(self, rgb, ms):

        return self.rgb_net(rgb) + self.ms_net(ms)

# =========================================================
# SEGFORMER
# =========================================================
class SegFormerModel(nn.Module):

    def __init__(self):

        super().__init__()

        config = SegformerConfig(num_labels=2)

        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, rgb):

        outputs = self.model(pixel_values=rgb)

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=rgb.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return logits

# =========================================================
# SAFE LOAD
# =========================================================
def safe_load_model(model, model_path):

    if os.path.exists(model_path):

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        if isinstance(checkpoint, dict):

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

        model.load_state_dict(
            checkpoint,
            strict=False
        )

        print(f"✅ Loaded: {model_path}")

    else:

        print(f"⚠️ Weight file not found: {model_path}")

# =========================================================
# COMPLEXITY FUNCTION
# =========================================================
def compute_complexity(model, model_name, use_ms=True):

    model.eval()

    rgb = torch.randn(1, 3, 512, 512).to(DEVICE)

    ms = torch.randn(1, 6, 512, 512).to(DEVICE)

    # =====================================================
    # PARAMS + FLOPS
    # =====================================================
    if use_ms:

        flops, params = profile(
            model,
            inputs=(rgb, ms),
            verbose=False
        )

    else:

        flops, params = profile(
            model,
            inputs=(rgb,),
            verbose=False
        )

    # =====================================================
    # FPS
    # =====================================================
    warmup_iters = 20
    test_iters = 100

    with torch.no_grad():

        # warmup
        for _ in range(warmup_iters):

            if use_ms:
                _ = model(rgb, ms)
            else:
                _ = model(rgb)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        for _ in range(test_iters):

            if use_ms:
                _ = model(rgb, ms)
            else:
                _ = model(rgb)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    fps = test_iters / (end - start)

    return {
        "Model": model_name,
        "Params": params / 1e6,
        "FLOPs": flops / 1e9,
        "FPS": fps
    }

# =========================================================
# MODEL PATHS
# =========================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

models = {

    "DenseUNet": {
        "model": LateFusionDenseUNet().to(DEVICE),
        "path": os.path.join(MODEL_DIR, "best_denseunet.pth"),
        "use_ms": True
    },

    "UNet++": {
        "model": LateFusionUNetPP().to(DEVICE),
        "path": os.path.join(MODEL_DIR, "best_unetpp_run0.pth"),
        "use_ms": True
    },

    "PSPNet": {
        "model": LateFusionPSPNet().to(DEVICE),
        "path": os.path.join(MODEL_DIR, "best_pspnet_run0.pth"),
        "use_ms": True
    },

    "SegFormer": {
        "model": SegFormerModel().to(DEVICE),
        "path": os.path.join(MODEL_DIR, "Segformer_run0.pth"),
        "use_ms": False
    },

    # =====================================================
    # ADD YOUR M-NET / DEEPLAB HERE
    # =====================================================
}

# =========================================================
# RUN
# =========================================================
results = []

print("\n" + "="*70)
print("COMPUTING PARAMS / FLOPS / FPS")
print("="*70)

for name, info in models.items():

    print(f"\nProcessing: {name}")

    model = info["model"]

    safe_load_model(model, info["path"])

    metrics = compute_complexity(
        model,
        name,
        use_ms=info["use_ms"]
    )

    results.append(metrics)

    print(f"Params : {metrics['Params']:.2f} M")
    print(f"FLOPs : {metrics['FLOPs']:.2f} G")
    print(f"FPS    : {metrics['FPS']:.2f}")

# =========================================================
# FINAL TABLE
# =========================================================
print("\n" + "="*70)
print(f"{'Model':<25} {'Params(M)':<15} {'FLOPs(G)':<15} {'FPS':<10}")
print("="*70)

for r in results:

    print(
        f"{r['Model']:<25} "
        f"{r['Params']:<15.2f} "
        f"{r['FLOPs']:<15.2f} "
        f"{r['FPS']:<10.2f}"
    )

print("="*70)