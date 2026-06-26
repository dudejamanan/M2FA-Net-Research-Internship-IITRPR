from thop import profile
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"

MODEL_PATH = "best_model_MultiLevelFusionDeepLab.pth"

SAVE_DIR = "mnet_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
class WeedyRiceDataset(Dataset):

    def __init__(self, root_dir, split_file):

        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.ms_dir = os.path.join(root_dir, "Multispectral")
        self.mask_dir = os.path.join(root_dir, "Masks")

        with open(split_file, "r") as f:

            self.samples = [
                x.strip().replace(".JPG","").replace(".jpg","")
                for x in f
            ]

        # APPROX NORMALIZATION STATS
        stats = np.load("ms_mean_std.npy", allow_pickle=True).item()

        self.ms_mean = torch.tensor(
            stats['mean'],
            dtype=torch.float32
        ).view(6,1,1)

        self.ms_std = torch.tensor(
            stats['std'],
            dtype=torch.float32
        ).view(6,1,1)
    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        base = self.samples[idx]

        # =====================================================
        # RGB
        # =====================================================
        rgb = cv2.imread(
            os.path.join(self.rgb_dir, base + ".JPG")
        )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, (512,512))

        # RGB IS 8-BIT
        rgb_np = rgb.astype(np.float32) / 255.0

        rgb_tensor = torch.tensor(
            np.transpose(rgb_np, (2,0,1)),
            dtype=torch.float32
        )

        # IMAGENET NORMALIZATION
        rgb_tensor = (
            rgb_tensor -
            torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        ) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)

        # =====================================================
        # MULTISPECTRAL
        # =====================================================
        bands = []

        for b in ["_G.TIF","_R.TIF","_RE.TIF","_NIR.TIF"]:

            img = cv2.imread(
                os.path.join(self.ms_dir, base+b),
                cv2.IMREAD_UNCHANGED
            )

            img = cv2.resize(img, (512,512))

            if img.ndim == 3:
                img = img[:,:,0]

            # TIFF IS 16-BIT
            img = img.astype(np.float32) / 65535.0

            bands.append(img)

        G, R, RE, NIR = bands

        eps = 1e-6

        NDVI = (NIR - R) / (NIR + R + eps)

        NDRE = (NIR - RE) / (NIR + RE + eps)

        NDVI = (NDVI + 1) / 2

        NDRE = (NDRE + 1) / 2

        # CHANNEL FIRST
        ms = np.stack(
            [G,R,RE,NIR,NDVI,NDRE],
            axis=0
        )

        ms_tensor = torch.tensor(
            ms,
            dtype=torch.float32
        )

        # NORMALIZATION
        ms_tensor = (
            ms_tensor - self.ms_mean
        ) / self.ms_std

        # =====================================================
        # MASK
        # =====================================================
        mask = cv2.imread(
            os.path.join(self.mask_dir, base + ".png"),
            0
        )

        mask = cv2.resize(mask, (512,512))

        mask = (mask > 0).astype(np.uint8)

        mask_tensor = torch.tensor(mask)

        return rgb_tensor, ms_tensor, mask_tensor, rgb_np

# =========================================================
# CBAM
# =========================================================
class CBAM(nn.Module):

    def __init__(self, channels, reduction=8):

        super().__init__()

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(2,1,7,padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x * self.ca(x)

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        x = x * self.sa(torch.cat([avg,mx], dim=1))

        return x
    
class MultiLevelFusionDeepLab(nn.Module):

    def __init__(self, use_cbam=True):

        super().__init__()

        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1

        self.rgb_backbone = deeplabv3_resnet50(
            weights=weights
        ).backbone

        self.ms_backbone = deeplabv3_resnet50(
            weights=weights
        ).backbone

        old_conv = self.ms_backbone.conv1

        self.ms_backbone.conv1 = nn.Conv2d(
            6,64,7,2,3,bias=False
        )

        # INITIALIZE EXTRA CHANNELS
        with torch.no_grad():

            mean_weight = old_conv.weight.mean(
                dim=1,
                keepdim=True
            )

            self.ms_backbone.conv1.weight[:, :3] = old_conv.weight

            self.ms_backbone.conv1.weight[:, 3:] = mean_weight.expand(
                -1,
                3,
                -1,
                -1
            )

        self.fusion1 = nn.Conv2d(512,256,1)
        self.fusion2 = nn.Conv2d(1024,512,1)
        self.fusion3 = nn.Conv2d(2048,1024,1)
        self.fusion4 = nn.Conv2d(4096,2048,1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(2048)

        self.use_cbam = use_cbam

        if use_cbam:

            self.attn1 = CBAM(256)
            self.attn2 = CBAM(512)
            self.attn3 = CBAM(1024)
            self.attn4 = CBAM(2048)

        self.classifier = nn.Sequential(
            nn.Conv2d(2048,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256,2,1)
        )

        self.aux_classifier1 = nn.Conv2d(256,2,1)
        self.aux_classifier2 = nn.Conv2d(512,2,1)
        self.aux_classifier3 = nn.Conv2d(1024,2,1)

    def extract_features(self, backbone, x):

        feats = []

        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        feats.append(x)

        x = backbone.layer2(x)
        feats.append(x)

        x = backbone.layer3(x)
        feats.append(x)

        x = backbone.layer4(x)
        feats.append(x)

        return feats

    def forward(self, rgb, ms):

        rgb_feats = self.extract_features(
            self.rgb_backbone,
            rgb
        )

        ms_feats = self.extract_features(
            self.ms_backbone,
            ms
        )

        fused1 = self.fusion1(
            torch.cat([rgb_feats[0], ms_feats[0]], dim=1)
        )

        fused1 = self.bn1(fused1)
        fused1 = F.relu(fused1)

        fused2 = self.fusion2(
            torch.cat([rgb_feats[1], ms_feats[1]], dim=1)
        )

        fused2 = self.bn2(fused2)
        fused2 = F.relu(fused2)

        fused3 = self.fusion3(
            torch.cat([rgb_feats[2], ms_feats[2]], dim=1)
        )

        fused3 = self.bn3(fused3)
        fused3 = F.relu(fused3)

        fused4 = self.fusion4(
            torch.cat([rgb_feats[3], ms_feats[3]], dim=1)
        )

        fused4 = self.bn4(fused4)
        fused4 = F.relu(fused4)

        if self.use_cbam:

            fused1 = self.attn1(fused1)
            fused2 = self.attn2(fused2)
            fused3 = self.attn3(fused3)
            fused4 = self.attn4(fused4)

        out = self.classifier(fused4)

        out = F.interpolate(
            out,
            size=rgb.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return out


model = MultiLevelFusionDeepLab().to(device)
model.eval()

rgb = torch.randn(1, 3, 512, 512).to(device)
ms  = torch.randn(1, 6, 512, 512).to(device)

flops, params = profile(model, inputs=(rgb, ms))

print(f"Params: {params/1e6:.2f} M")
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")

import time
import torch

model.eval()

rgb = torch.randn(1, 3, 512, 512).to(device)
ms  = torch.randn(1, 6, 512, 512).to(device)

# Warmup
for _ in range(20):
    with torch.no_grad():
        _ = model(rgb, ms)

torch.cuda.synchronize()

# Timing
start = time.time()

num_iters = 100

for _ in range(num_iters):
    with torch.no_grad():
        _ = model(rgb, ms)

torch.cuda.synchronize()

end = time.time()

fps = num_iters / (end - start)

print(f"FPS: {fps:.2f}")