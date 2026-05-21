import os
import cv2
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd
from datetime import datetime

# ====================== Reproducibility ======================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ====================== Combined Loss Function ======================
# class CombinedLoss(nn.Module):
#     def __init__(self, weight_ce=0.4, weight_dice=0.4):
#         super().__init__()
#         self.weight_ce = weight_ce
#         self.weight_dice = weight_dice
        
#     def forward(self, preds, targets):
#         ce_loss = F.cross_entropy(preds, targets)
        
#         preds_softmax = F.softmax(preds, dim=1)
#         preds_dice = preds_softmax[:, 1, :, :]
#         targets_dice = (targets == 1).float()
        
#         smooth = 1.0
#         intersection = (preds_dice * targets_dice).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (preds_dice.sum() + targets_dice.sum() + smooth)
        
#         total_loss = (self.weight_ce * ce_loss + self.weight_dice * dice_loss)
#         return total_loss

class CombinedLoss(nn.Module):
    """
    Combined Loss: Cross Entropy + Dice Loss + Boundary Loss
    """
    def __init__(self, weight_ce=0.4, weight_dice=0.4, weight_boundary=0.2):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        
    def forward(self, preds, targets):
        """
        Args:
            preds: (B, 2, H, W) - logits
            targets: (B, H, W) - long tensor
        """
        # 1. Cross Entropy Loss
        ce_loss = F.cross_entropy(preds, targets)
        
        # 2. Dice Loss (focuses on foreground class - index 1)
        preds_softmax = F.softmax(preds, dim=1)
        preds_dice = preds_softmax[:, 1, :, :]  # Probability of foreground
        targets_dice = (targets == 1).float()   # Binary mask
        
        # Smooth factor to avoid division by zero
        smooth = 1.0
        intersection = (preds_dice * targets_dice).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (preds_dice.sum() + targets_dice.sum() + smooth)
        
        # 3. Boundary Loss using Laplacian kernel
        # Create Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]]]], 
                                       device=preds.device, 
                                       dtype=torch.float32)
        
        # # Apply Laplacian to predictions and targets
        pred_edges = F.conv2d(preds_dice.unsqueeze(1), laplacian_kernel, padding=1)
        target_edges = F.conv2d(targets_dice.unsqueeze(1), laplacian_kernel, padding=1)
        
        # # Boundary loss is L1 difference between edge maps
        boundary_loss = F.l1_loss(pred_edges, target_edges)
        
        # # Combined loss
        total_loss = (self.weight_ce * ce_loss + 
                     self.weight_dice * dice_loss + 
                     self.weight_boundary * boundary_loss)
        
        # total_loss = (self.weight_ce * ce_loss + 
        #              self.weight_dice * dice_loss)
        
        return total_loss
# ====================== Warmup Cosine Scheduler ======================
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * warmup_factor for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
            return [lr for _ in self.base_lrs]

# ====================== No Warmup Scheduler (for ablation) ======================
class NoWarmupScheduler:
    """Constant learning rate (no warmup, no cosine decay)"""
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
    
    def step(self):
        pass
    
    def get_last_lr(self):
        return [self.lr]

# ====================== Metrics ======================
def compute_metrics(preds, masks):
    # Handle tuple output (for models that return aux outputs)
    if isinstance(preds, tuple):
        preds = preds[0]  # Take the main prediction
    
    preds = torch.argmax(preds, dim=1).view(-1)
    masks = masks.view(-1)

    TP = ((preds == 1) & (masks == 1)).sum().item()
    FP = ((preds == 1) & (masks == 0)).sum().item()
    FN = ((preds == 0) & (masks == 1)).sum().item()
    TN = ((preds == 0) & (masks == 0)).sum().item()

    iou = TP / (TP + FP + FN + 1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    
    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }

# ====================== Evaluation ======================
@torch.no_grad()
def evaluate(model, loader, device, name="Eval"):
    model.eval()
    metrics_sum = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0}
    count = 0

    for rgb, ms, masks in loader:
        rgb, ms, masks = rgb.to(device), ms.to(device), masks.to(device)

        with torch.amp.autocast(device_type=device.type):
            output = model(rgb, ms)
            # Handle tuple output
            if isinstance(output, tuple):
                output = output[0]

        metrics = compute_metrics(output, masks)
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]
        count += 1

    results = {key: value / count for key, value in metrics_sum.items()}
    print(f"{name} → IoU: {results['IoU']:.4f}, Dice: {results['Dice']:.4f}, "
          f"F1: {results['F1']:.4f}, Precision: {results['Precision']:.4f}, Recall: {results['Recall']:.4f}")
    return results

# ====================== MS Statistics ======================
def compute_ms_statistics(root_dir, split_file):
    ms_dir = os.path.join(root_dir, "Multispectral")
    with open(split_file, "r") as f:
        files = [x.strip().replace(".JPG", "").replace(".jpg", "") for x in f]

    means = np.zeros(6)
    stds = np.zeros(6)
    count = 0

    for base in tqdm(files, desc="Computing MS statistics"):
        bands = []
        for b in ["_G.TIF", "_R.TIF", "_RE.TIF", "_NIR.TIF"]:
            path = os.path.join(ms_dir, base + b)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = cv2.resize(img, (512, 512))
            if img.ndim == 3:
                img = img[:, :, 0]
            bands.append(img.astype(np.float32) / 65535.0)

        if len(bands) != 4:
            continue

        G, R, RE, NIR = bands
        NDVI = (NIR - R) / (NIR + R + 1e-6)
        NDRE = (NIR - RE) / (NIR + RE + 1e-6)
        NDVI = (NDVI + 1) / 2
        NDRE = (NDRE + 1) / 2
        ms_stack = np.stack([G, R, RE, NIR, NDVI, NDRE], axis=-1)

        means += ms_stack.mean(axis=(0, 1))
        stds += ms_stack.std(axis=(0, 1))
        count += 1

    means /= count
    stds /= count
    np.save("ms_mean_std.npy", {"mean": means, "std": stds})

# ====================== Augmentation ======================
class SimpleAugmentation:
    """Simple flips augmentation (horizontal + vertical)"""
    def __call__(self, rgb, ms, mask):
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            ms = np.fliplr(ms).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            rgb = np.flipud(rgb).copy()
            ms = np.flipud(ms).copy()
            mask = np.flipud(mask).copy()
        return rgb, ms, mask

# ====================== Dataset ======================
class WeedyRiceDataset(Dataset):
    def __init__(self, root_dir, split_file, augment=False):
        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.ms_dir = os.path.join(root_dir, "Multispectral")
        self.mask_dir = os.path.join(root_dir, "Masks")
        self.augment = augment

        with open(split_file, "r") as f:
            self.samples = [x.strip().replace(".JPG", "").replace(".jpg", "") for x in f]

        if not os.path.exists("ms_mean_std.npy"):
            compute_ms_statistics(root_dir, split_file)

        stats = np.load("ms_mean_std.npy", allow_pickle=True).item()
        self.ms_mean = torch.tensor(stats['mean'], dtype=torch.float32).view(6,1,1)
        self.ms_std  = torch.tensor(stats['std'], dtype=torch.float32).view(6,1,1)

        if augment:
            self.augmentation = SimpleAugmentation()
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        rgb = cv2.imread(os.path.join(self.rgb_dir, base + ".JPG"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512))

        bands = []
        for b in ["_G.TIF", "_R.TIF", "_RE.TIF", "_NIR.TIF"]:
            img = cv2.imread(os.path.join(self.ms_dir, base + b), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (512, 512))
            if img.ndim == 3:
                img = img[:, :, 0]
            bands.append(img.astype(np.float32) / 65535.0)

        G, R, RE, NIR = bands
        eps = 1e-6
        NDVI = (NIR - R) / (NIR + R + eps)
        NDRE = (NIR - RE) / (NIR + RE + eps)
        NDVI = (NDVI + 1) / 2
        NDRE = (NDRE + 1) / 2
        ms_np = np.stack([G, R, RE, NIR, NDVI, NDRE], axis=-1)

        mask = cv2.imread(os.path.join(self.mask_dir, base + ".png"), 0)
        mask = cv2.resize(mask, (512, 512))
        mask = (mask > 0).astype(np.uint8)

        if self.augmentation is not None:
            rgb_np = rgb.astype(np.float32) / 255.0
            ms_np = ms_np.astype(np.float32)
            mask_np = mask.astype(np.uint8)
            rgb_np, ms_np, mask_np = self.augmentation(rgb_np, ms_np, mask_np)
            rgb = (rgb_np * 255).astype(np.uint8)
            ms_np = ms_np
            mask = mask_np
        
        rgb = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        ms = torch.from_numpy(ms_np).permute(2,0,1).float()

        rgb = (rgb - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / \
              torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        ms = (ms - self.ms_mean) / self.ms_std

        return rgb, ms, torch.from_numpy(mask).long()

# ====================== CBAM Attention ======================
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
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg, mx], dim=1))
        return x

# ====================== FULL MODEL (Base Class) ======================
class MultiLevelFusionDeepLab(nn.Module):
    """Full model: Multi-level fusion + CBAM + Deep Supervision"""
    def __init__(self, use_cbam=True):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        
        self.rgb_backbone = deeplabv3_resnet50(weights=weights).backbone
        self.ms_backbone = deeplabv3_resnet50(weights=weights).backbone
        
        old_conv = self.ms_backbone.conv1
        self.ms_backbone.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            self.ms_backbone.conv1.weight[:, :3] = old_conv.weight
            self.ms_backbone.conv1.weight[:, 3:] = mean_weight.expand(-1, 3, -1, -1)
        
        self.fusion1 = nn.Conv2d(256 * 2, 256, 1)
        self.fusion2 = nn.Conv2d(512 * 2, 512, 1)
        self.fusion3 = nn.Conv2d(1024 * 2, 1024, 1)
        self.fusion4 = nn.Conv2d(2048 * 2, 2048, 1)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.attn1 = CBAM(256)
            self.attn2 = CBAM(512)
            self.attn3 = CBAM(1024)
            self.attn4 = CBAM(2048)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(2048)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 2, 1)
        )
        
        self.aux_classifier1 = nn.Conv2d(256, 2, 1)
        self.aux_classifier2 = nn.Conv2d(512, 2, 1)
        self.aux_classifier3 = nn.Conv2d(1024, 2, 1)
        
    def forward(self, rgb, ms):
        rgb_features = []
        ms_features = []
        
        def extract_features(backbone, x, features_list):
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x); features_list.append(x)
            x = backbone.layer2(x); features_list.append(x)
            x = backbone.layer3(x); features_list.append(x)
            x = backbone.layer4(x); features_list.append(x)
            return x
        
        rgb_final = extract_features(self.rgb_backbone, rgb, rgb_features)
        ms_final = extract_features(self.ms_backbone, ms, ms_features)
        
        fused1 = torch.cat([rgb_features[0], ms_features[0]], dim=1)
        fused1 = self.fusion1(fused1)
        fused1 = self.bn1(fused1)
        fused1 = F.relu(fused1)
        if self.use_cbam:
            fused1 = self.attn1(fused1)
        
        fused2 = torch.cat([rgb_features[1], ms_features[1]], dim=1)
        fused2 = self.fusion2(fused2)
        fused2 = self.bn2(fused2)
        fused2 = F.relu(fused2)
        if self.use_cbam:
            fused2 = self.attn2(fused2)
        
        fused3 = torch.cat([rgb_features[2], ms_features[2]], dim=1)
        fused3 = self.fusion3(fused3)
        fused3 = self.bn3(fused3)
        fused3 = F.relu(fused3)
        if self.use_cbam:
            fused3 = self.attn3(fused3)
        
        fused4 = torch.cat([rgb_features[3], ms_features[3]], dim=1)
        fused4 = self.fusion4(fused4)
        fused4 = self.bn4(fused4)
        fused4 = F.relu(fused4)
        if self.use_cbam:
            fused4 = self.attn4(fused4)
        
        x = self.classifier(fused4)
        
        aux_outputs = []
        if self.training:
            aux1 = self.aux_classifier1(fused1)
            aux2 = self.aux_classifier2(fused2)
            aux3 = self.aux_classifier3(fused3)
            aux_outputs = [aux1, aux2, aux3]
        
        x = F.interpolate(x, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.training and aux_outputs:
            aux_upsampled = [F.interpolate(aux, size=rgb.shape[-2:], mode='bilinear', align_corners=False) 
                           for aux in aux_outputs]
            return x, aux_upsampled
        else:
            return x

# ====================== BASELINE: Early Fusion ======================
class EarlyFusionDeepLab(nn.Module):
    """Baseline: Simple early fusion of RGB + MS"""
    def __init__(self):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        deeplab = deeplabv3_resnet50(weights=weights)
        
        old_conv = deeplab.backbone.conv1
        deeplab.backbone.conv1 = nn.Conv2d(9, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            deeplab.backbone.conv1.weight[:, :3] = old_conv.weight
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            deeplab.backbone.conv1.weight[:, 3:] = mean_weight.expand(-1, 6, -1, -1)
        
        self.backbone = deeplab.backbone
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )
    
    def forward(self, rgb, ms):
        x = torch.cat([rgb, ms], dim=1)
        features = self.backbone(x)
        x = self.classifier(features['out'])
        x = F.interpolate(x, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        return x

# ====================== ABLATION 1: No Multi-Level Fusion ======================
class SingleLevelFusion(nn.Module):
    """Ablation: Only fuse at the final level"""
    def __init__(self):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.rgb_backbone = deeplabv3_resnet50(weights=weights).backbone
        self.ms_backbone = deeplabv3_resnet50(weights=weights).backbone
        
        old_conv = self.ms_backbone.conv1
        self.ms_backbone.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            self.ms_backbone.conv1.weight[:, :3] = old_conv.weight
            self.ms_backbone.conv1.weight[:, 3:] = mean_weight.expand(-1, 3, -1, -1)
        
        self.final_fusion = nn.Conv2d(2048 * 2, 2048, 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)
        )
    
    def forward(self, rgb, ms):
        def extract_final(backbone, x):
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            return x
        
        rgb_final = extract_final(self.rgb_backbone, rgb)
        ms_final = extract_final(self.ms_backbone, ms)
        
        fused = torch.cat([rgb_final, ms_final], dim=1)
        fused = self.final_fusion(fused)
        x = self.classifier(fused)
        x = F.interpolate(x, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        return x

# ====================== ABLATION 2: No CBAM Attention ======================
class NoAttentionFusion(MultiLevelFusionDeepLab):
    """Ablation: Multi-level fusion without CBAM attention"""
    def __init__(self):
        super().__init__(use_cbam=False)

# ====================== ABLATION 3: No Deep Supervision ======================
class NoDeepSupervisionFusion(nn.Module):
    """Ablation: Multi-level fusion without auxiliary losses"""
    def __init__(self):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        
        self.rgb_backbone = deeplabv3_resnet50(weights=weights).backbone
        self.ms_backbone = deeplabv3_resnet50(weights=weights).backbone
        
        old_conv = self.ms_backbone.conv1
        self.ms_backbone.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            self.ms_backbone.conv1.weight[:, :3] = old_conv.weight
            self.ms_backbone.conv1.weight[:, 3:] = mean_weight.expand(-1, 3, -1, -1)
        
        self.fusion1 = nn.Conv2d(256 * 2, 256, 1)
        self.fusion2 = nn.Conv2d(512 * 2, 512, 1)
        self.fusion3 = nn.Conv2d(1024 * 2, 1024, 1)
        self.fusion4 = nn.Conv2d(2048 * 2, 2048, 1)
        
        self.attn1 = CBAM(256)
        self.attn2 = CBAM(512)
        self.attn3 = CBAM(1024)
        self.attn4 = CBAM(2048)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(2048)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 2, 1)
        )
        
    def forward(self, rgb, ms):
        rgb_features = []
        ms_features = []
        
        def extract_features(backbone, x, features_list):
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x); features_list.append(x)
            x = backbone.layer2(x); features_list.append(x)
            x = backbone.layer3(x); features_list.append(x)
            x = backbone.layer4(x); features_list.append(x)
            return x
        
        rgb_final = extract_features(self.rgb_backbone, rgb, rgb_features)
        ms_final = extract_features(self.ms_backbone, ms, ms_features)
        
        fused1 = torch.cat([rgb_features[0], ms_features[0]], dim=1)
        fused1 = self.fusion1(fused1)
        fused1 = self.bn1(fused1)
        fused1 = F.relu(fused1)
        fused1 = self.attn1(fused1)
        
        fused2 = torch.cat([rgb_features[1], ms_features[1]], dim=1)
        fused2 = self.fusion2(fused2)
        fused2 = self.bn2(fused2)
        fused2 = F.relu(fused2)
        fused2 = self.attn2(fused2)
        
        fused3 = torch.cat([rgb_features[2], ms_features[2]], dim=1)
        fused3 = self.fusion3(fused3)
        fused3 = self.bn3(fused3)
        fused3 = F.relu(fused3)
        fused3 = self.attn3(fused3)
        
        fused4 = torch.cat([rgb_features[3], ms_features[3]], dim=1)
        fused4 = self.fusion4(fused4)
        fused4 = self.bn4(fused4)
        fused4 = F.relu(fused4)
        fused4 = self.attn4(fused4)
        
        x = self.classifier(fused4)
        x = F.interpolate(x, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        return x

# ====================== Training Function ======================
def train_model(model, train_loader, val_loader, device, config, use_warmup=True):
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config['base_lr'], 
                                  weight_decay=config['weight_decay'])
    
    # criterion = CombinedLoss(weight_ce=0.4, weight_dice=0.4)
    criterion = CombinedLoss(weight_ce=0.4, weight_dice=0.4, weight_boundary=0.2)
    
    if use_warmup:
        scheduler = WarmupCosineScheduler(optimizer, 
                                         warmup_epochs=config['warmup_epochs'],
                                         total_epochs=config['num_epochs'],
                                         base_lr=config['base_lr'],
                                         min_lr=config['min_lr'])
    else:
        scheduler = NoWarmupScheduler(optimizer, config['base_lr'])
    
    scaler = torch.cuda.amp.GradScaler() if config['use_mixed_precision'] else None
    
    best_iou = 0
    best_metrics = None
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for rgb, ms, mask in progress_bar:
            rgb, ms, mask = rgb.to(device), ms.to(device), mask.to(device)
            optimizer.zero_grad()
            
            if config['use_mixed_precision'] and scaler:
                with torch.cuda.amp.autocast():
                    output = model(rgb, ms)
                    if isinstance(output, tuple) and len(output) == 2:
                        preds, aux_preds = output
                        loss = criterion(preds, mask)
                        if aux_preds:
                            aux_loss = sum(criterion(aux, mask) for aux in aux_preds)
                            loss += 0.3 * aux_loss
                    else:
                        preds = output
                        loss = criterion(preds, mask)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(rgb, ms)
                if isinstance(output, tuple) and len(output) == 2:
                    preds, aux_preds = output
                    loss = criterion(preds, mask)
                    if aux_preds:
                        aux_loss = sum(criterion(aux, mask) for aux in aux_preds)
                        loss += 0.3 * aux_loss
                else:
                    preds = output
                    loss = criterion(preds, mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, name=f"Val Epoch {epoch+1}")
        
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            best_metrics = val_metrics
    
    return best_iou, best_metrics

# ====================== SAVE PREDICTIONS FOR ABLATION ======================
@torch.no_grad()
def save_ablation_predictions(
    model,
    dataset,
    device,
    save_dir,
    variant_name,
    sample_indices=[1, 5, 10, 15, 20]
):
    """
    Save predictions for fixed sample indices.
    """

    print(f"\n🎨 Saving predictions for: {variant_name}")

    model.eval()

    # create folder name safely
    variant_folder = os.path.join(
        save_dir,
        variant_name.replace(" ", "_").replace(":", "")
    )

    os.makedirs(variant_folder, exist_ok=True)

    fig, axes = plt.subplots(
        len(sample_indices),
        3,
        figsize=(14, 4 * len(sample_indices))
    )

    if len(sample_indices) == 1:
        axes = [axes]

    for row, idx in enumerate(sample_indices):

        # ======================
        # LOAD SAMPLE
        # ======================
        rgb, ms, mask = dataset[idx]

        # ======================
        # RGB VISUALIZATION
        # ======================
        rgb_vis = rgb.clone()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        rgb_vis = rgb_vis * std + mean

        rgb_vis = rgb_vis.permute(1,2,0).cpu().numpy()
        rgb_vis = np.clip(rgb_vis, 0, 1)

        # ======================
        # MODEL INPUT
        # ======================
        rgb_input = rgb.unsqueeze(0).to(device)
        ms_input  = ms.unsqueeze(0).to(device)

        # ======================
        # PREDICTION
        # ======================
        with torch.no_grad():

            if device.type == "cuda":

                with torch.cuda.amp.autocast():

                    output = model(rgb_input, ms_input)

            else:

                output = model(rgb_input, ms_input)

        # handle tuple output
        if isinstance(output, tuple):

            if len(output) == 2:
                preds = output[0]
            else:
                preds = output

        else:
            preds = output

        pred_mask = torch.argmax(
            preds,
            dim=1
        ).squeeze(0).cpu().numpy()

        gt_mask = mask.cpu().numpy()

        # ======================
        # RGB
        # ======================
        axes[row, 0].imshow(rgb_vis)
        axes[row, 0].set_title(f"RGB Image ({idx})")
        axes[row, 0].axis("off")

        # ======================
        # GROUND TRUTH
        # ======================
        axes[row, 1].imshow(gt_mask, cmap='gray')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        # ======================
        # PREDICTION
        # ======================
        axes[row, 2].imshow(pred_mask, cmap='gray')
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        variant_folder,
        "sample_predictions.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"✅ Saved predictions to: {save_path}")

# ====================== ABLATION STUDY MAIN ======================
def run_ablation_study():
    """Run complete ablation study with multiple runs - 20 epochs"""
    
    root = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"ablation_study_results_20epochs_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    NUM_RUNS = 1 # Number of runs for statistical significance
    NUM_EPOCHS = 20  # ← CHANGED TO 20 EPOCHS (matches original)
    
    # Define all model variants for ablation
    variants = {
        'Baseline (Early Fusion)': {
            'model_class': EarlyFusionDeepLab,
            'model_kwargs': {},
            'use_warmup': True,
            'use_augmentation': False,
            'description': 'Simple early fusion of RGB+MS (9 channels)'
        },
        'Ablation: No Multi-Level Fusion': {
            'model_class': SingleLevelFusion,
            'model_kwargs': {},
            'use_warmup': True,
            'use_augmentation': False,
            'description': 'Only final-level fusion, no per-level fusion'
        },
        'Ablation: No CBAM Attention': {
            'model_class': NoAttentionFusion,
            'model_kwargs': {},
            'use_warmup': True,
            'use_augmentation': False,
            'description': 'Multi-level fusion without CBAM attention'
        },
        'Ablation: No Deep Supervision': {
            'model_class': NoDeepSupervisionFusion,
            'model_kwargs': {},
            'use_warmup': True,
            'use_augmentation': False,
            'description': 'Multi-level fusion without auxiliary losses'
        },
        'Ablation: No Warmup Scheduler': {
            'model_class': MultiLevelFusionDeepLab,
            'model_kwargs': {'use_cbam': True},
            'use_warmup': False,
            'use_augmentation': False,
            'description': 'Multi-level fusion with constant LR (no warmup/cosine)'
        },
        'Ablation: No Augmentation': {
            'model_class': MultiLevelFusionDeepLab,
            'model_kwargs': {'use_cbam': True},
            'use_warmup': True,
            'use_augmentation': False,  # ← NO AUGMENTATION
            'description': 'Full model without data augmentation'
        },
        'FULL MODEL (Proposed)': {
            'model_class': MultiLevelFusionDeepLab,
            'model_kwargs': {'use_cbam': True},
            'use_warmup': True,
            'use_augmentation': True,
            'description': 'Multi-level fusion + CBAM + Deep Supervision + Warmup Cosine + Augmentation'
        }
    }
    
    # Prepare datasets
    print("\n" + "="*80)
    print("PREPARING DATASETS")
    print("="*80)
    
    # We'll create datasets inside the loop based on augmentation flag
    val_dataset = WeedyRiceDataset(root, "val_list.txt", augment=False)
    test_dataset = WeedyRiceDataset(root, "test_list.txt", augment=False)
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    config = {
        'base_lr': 5e-5,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'warmup_epochs': 3,
        'num_epochs': NUM_EPOCHS,
        'batch_size': 4,
        'use_mixed_precision': torch.cuda.is_available(),
    }
    
    print(f"\n📊 Training Configuration:")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Runs per variant: {NUM_RUNS}")
    print(f"   - Learning rate: {config['base_lr']}")
    print(f"   - Warmup epochs: {config['warmup_epochs']}")
    
    # Store results
    all_results = {}
    
    print("\n" + "="*80)
    print("STARTING ABLATION STUDY")
    print(f"Each variant will be run {NUM_RUNS} times for statistical significance")
    print("="*80)
    
    for variant_name, variant_info in variants.items():
        print("\n" + "="*80)
        print(f"📊 TESTING: {variant_name}")
        print(f"   {variant_info['description']}")
        print(f"   Augmentation: {variant_info.get('use_augmentation', True)}")
        print("="*80)
        
        run_results = []
        
        for run in range(NUM_RUNS):
            print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
            set_seed(42 + run)
            
            # Create dataset with appropriate augmentation
            use_aug = variant_info.get('use_augmentation', True)
            train_dataset = WeedyRiceDataset(root, "train_list.txt", augment=use_aug)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
            
            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Augmentation: {'ON' if use_aug else 'OFF'}")
            
            # Create model
            model = variant_info['model_class'](**variant_info['model_kwargs']).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Train model
            best_iou, val_metrics = train_model(
                model, train_loader, val_loader, device, config, 
                use_warmup=variant_info['use_warmup']
            )
            
            # Evaluate on test set
            test_metrics = evaluate(model, test_loader, device, name=f"Test Run {run+1}")

            # ======================
            # SAVE SAMPLE PREDICTIONS
            # ======================
            save_ablation_predictions(
                model=model,
                dataset=val_dataset,
                device=device,
                save_dir=save_dir,
                variant_name=f"{variant_name}_Run_{run+1}"
)
            
            run_results.append({
                'run': run + 1,
                'val_iou': val_metrics['IoU'],
                'val_dice': val_metrics['Dice'],
                'test_iou': test_metrics['IoU'],
                'test_dice': test_metrics['Dice'],
                'test_precision': test_metrics['Precision'],
                'test_recall': test_metrics['Recall'],
                'test_f1': test_metrics['F1'],
                'params': total_params
            })
        
        # Calculate statistics
        test_ious = [r['test_iou'] for r in run_results]
        test_dices = [r['test_dice'] for r in run_results]
        
        all_results[variant_name] = {
            'runs': run_results,
            'mean_iou': np.mean(test_ious),
            'std_iou': np.std(test_ious),
            'mean_dice': np.mean(test_dices),
            'std_dice': np.std(test_dices),
            'params': run_results[0]['params']
        }
        
        print(f"\n✅ {variant_name} Summary: IoU = {all_results[variant_name]['mean_iou']:.4f} ± {all_results[variant_name]['std_iou']:.4f}, "
              f"Dice = {all_results[variant_name]['mean_dice']:.4f} ± {all_results[variant_name]['std_dice']:.4f}")
    
    # ====================== PRINT FINAL TABLE ======================
    print("\n" + "="*120)
    print("                     ABLATION STUDY RESULTS (20 Epochs)")
    print("="*120)
    print(f"{'Variant':<35} {'Params (M)':<12} {'Test IoU':<20} {'Test Dice':<20} {'Δ from Full':<15}")
    print("-"*120)
    
    full_model_iou = all_results['FULL MODEL (Proposed)']['mean_iou']
    full_model_dice = all_results['FULL MODEL (Proposed)']['mean_dice']
    
    for variant_name, results in all_results.items():
        params_m = results['params'] / 1e6
        iou_str = f"{results['mean_iou']:.4f} ± {results['std_iou']:.4f}"
        dice_str = f"{results['mean_dice']:.4f} ± {results['std_dice']:.4f}"
        
        if variant_name == 'FULL MODEL (Proposed)':
            delta_iou = "---"
            delta_dice = "---"
        else:
            delta_iou = f"-{(full_model_iou - results['mean_iou']) * 100:.2f}%"
            delta_dice = f"-{(full_model_dice - results['mean_dice']) * 100:.2f}%"
        
        # Highlight the augmentation row
        if 'No Augmentation' in variant_name:
            print(f"\033[93m{variant_name:<35} {params_m:<12.2f} {iou_str:<20} {dice_str:<20} {delta_iou:<15}\033[0m")
        else:
            print(f"{variant_name:<35} {params_m:<12.2f} {iou_str:<20} {dice_str:<20} {delta_iou:<15}")
    
    print("="*120)
    
    # ====================== SAVE RESULTS ======================
    
    
    # Save as CSV
    df_rows = []
    for variant_name, results in all_results.items():
        for run in results['runs']:
            df_rows.append({
                'Variant': variant_name,
                'Run': run['run'],
                'Augmentation': 'Yes' if 'No Augmentation' not in variant_name or variant_name == 'FULL MODEL (Proposed)' else ('No' if 'No Augmentation' in variant_name else 'Yes'),
                'Val_IoU': run['val_iou'],
                'Val_Dice': run['val_dice'],
                'Test_IoU': run['test_iou'],
                'Test_Dice': run['test_dice'],
                'Test_Precision': run['test_precision'],
                'Test_Recall': run['test_recall'],
                'Test_F1': run['test_f1'],
                'Parameters': run['params']
            })
    
    df = pd.DataFrame(df_rows)
    df.to_csv(os.path.join(save_dir, 'ablation_results.csv'), index=False)
    
    # Save summary
    summary_rows = []
    for variant_name, results in all_results.items():
        summary_rows.append({
            'Variant': variant_name,
            'Mean_IoU': results['mean_iou'],
            'Std_IoU': results['std_iou'],
            'Mean_Dice': results['mean_dice'],
            'Std_Dice': results['std_dice'],
            'Parameters_M': results['params'] / 1e6
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, 'ablation_summary.csv'), index=False)
    
    # Create bar plot
    plt.figure(figsize=(16, 6))
    
    variants_list = list(all_results.keys())
    iou_means = [all_results[v]['mean_iou'] for v in variants_list]
    iou_stds = [all_results[v]['std_iou'] for v in variants_list]
    dice_means = [all_results[v]['mean_dice'] for v in variants_list]
    dice_stds = [all_results[v]['std_dice'] for v in variants_list]
    
    # Color code: highlight augmentation row
    colors_iou = ['#ff9999' if 'No Augmentation' in v else '#66b3ff' for v in variants_list]
    colors_dice = ['#ff9999' if 'No Augmentation' in v else '#99ff99' for v in variants_list]
    
    x = np.arange(len(variants_list))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x, iou_means, width, yerr=iou_stds, capsize=5, color=colors_iou, edgecolor='black')
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title('Ablation Study: IoU Comparison (20 epochs)', fontsize=14)
    plt.xticks(x, [v[:25] + '...' if len(v) > 25 else v for v in variants_list], rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.70, 0.80)
    
    # Add value labels
    for bar, val in zip(bars1, iou_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x, dice_means, width, yerr=dice_stds, capsize=5, color=colors_dice, edgecolor='black')
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title('Ablation Study: Dice Comparison (20 epochs)', fontsize=14)
    plt.xticks(x, [v[:25] + '...' if len(v) > 25 else v for v in variants_list], rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.85, 0.89)
    
    # Add value labels
    for bar, val in zip(bars2, dice_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Results saved to: {save_dir}")
    print(f"   - ablation_results.csv (detailed per-run results)")
    print(f"   - ablation_summary.csv (summary statistics)")
    print(f"   - ablation_results.png (visualization)")
    
    # Calculate augmentation benefit
    if 'FULL MODEL (Proposed)' in all_results and 'Ablation: No Augmentation' in all_results:
        aug_gain_iou = (all_results['FULL MODEL (Proposed)']['mean_iou'] - 
                       all_results['Ablation: No Augmentation']['mean_iou']) * 100
        aug_gain_dice = (all_results['FULL MODEL (Proposed)']['mean_dice'] - 
                        all_results['Ablation: No Augmentation']['mean_dice']) * 100
        print(f"\n📈 Augmentation Benefit: +{aug_gain_iou:.2f}% IoU, +{aug_gain_dice:.2f}% Dice")
    
    return all_results, save_dir

# ====================== MAIN ======================
if __name__ == "__main__":
    print("="*80)
    print("              ABLATION STUDY FOR WEED SEGMENTATION")
    print("="*80)
    print("\nThis script will run multiple ablation experiments to evaluate:")
    print("  1. Baseline: Early Fusion (RGB+MS concatenation)")
    print("  2. Ablation: No Multi-Level Fusion")
    print("  3. Ablation: No CBAM Attention")
    print("  4. Ablation: No Deep Supervision")
    print("  5. Ablation: No Warmup Scheduler")
    print("  6. Ablation: No Augmentation")  # ← NEW
    print("  7. FULL MODEL: All components together")
    print("\n✅ Each variant will be run 3 times for statistical significance")
    print("✅ Training for 20 epochs (matching your original training)")
    print("="*80)
    
    results, save_dir = run_ablation_study()
    
    print("\n" + "="*80)
    print("                     ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"\n📊 Key Findings:")
    
    if 'FULL MODEL (Proposed)' in results and 'Baseline (Early Fusion)' in results:
        full_model = results['FULL MODEL (Proposed)']
        baseline = results['Baseline (Early Fusion)']
        
        improvement_iou = (full_model['mean_iou'] - baseline['mean_iou']) * 100
        improvement_dice = (full_model['mean_dice'] - baseline['mean_dice']) * 100
        
        print(f"   • Full model improves IoU by {improvement_iou:.2f}% over baseline")
        print(f"   • Full model improves Dice by {improvement_dice:.2f}% over baseline")
    
    if 'Ablation: No Augmentation' in results:
        no_aug = results['Ablation: No Augmentation']
        full = results['FULL MODEL (Proposed)']
        aug_gain_iou = (full['mean_iou'] - no_aug['mean_iou']) * 100
        aug_gain_dice = (full['mean_dice'] - no_aug['mean_dice']) * 100
        print(f"   • Data augmentation provides +{aug_gain_iou:.2f}% IoU and +{aug_gain_dice:.2f}% Dice")
    
    print(f"\n📁 Results saved to: {save_dir}")
    print("\n📝 For your paper, report:")
    print("   - Mean ± Std over 3 runs")
    print("   - Percentage improvement from each component")
    print("   - Separate row for augmentation contribution")
    print("="*80)