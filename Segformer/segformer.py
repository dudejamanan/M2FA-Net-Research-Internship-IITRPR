import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
import csv
from torchvision.models.segmentation import deeplabv3_resnet50
# from transformers import SegformerForSemanticSegmentation
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(preds, masks):
    preds = torch.argmax(preds, dim=1)

    preds = preds.view(-1)
    masks = masks.view(-1)

    TP = ((preds == 1) & (masks == 1)).sum().item()
    FP = ((preds == 1) & (masks == 0)).sum().item()
    FN = ((preds == 0) & (masks == 1)).sum().item()
    TN = ((preds == 0) & (masks == 0)).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    return accuracy, precision, recall, iou, dice

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device, exp_type):
    model.eval()
    total = {"Accuracy":0,"Precision":0,"Recall":0,"IoU":0,"Dice":0}
    count = 0

    with torch.no_grad():
        for rgb, ms, masks in loader:
            rgb, ms, masks = rgb.to(device), ms.to(device), masks.to(device)

            if exp_type == "segformer":
                preds = model(rgb)
            else:
                preds, _, _, _ = model(rgb, ms)

            preds = F.interpolate(preds, size=masks.shape[1:], mode='bilinear', align_corners=False)

            acc, prec, rec, iou, dice = compute_metrics(preds, masks)

            total["Accuracy"] += acc
            total["Precision"] += prec
            total["Recall"] += rec
            total["IoU"] += iou
            total["Dice"] += dice
            count += 1

    return {k: v/count for k,v in total.items()}

# -----------------------------
# Dataset
# -----------------------------
class WeedyRiceDataset(Dataset):
    def __init__(self, root_dir, split_file, use_indices=True):
        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.ms_dir = os.path.join(root_dir, "Multispectral")
        self.mask_dir = os.path.join(root_dir, "Masks")
        self.use_indices = use_indices

        with open(split_file, "r") as f:
            files = [x.strip().replace(".JPG","").replace(".jpg","") for x in f]

        self.samples = self._filter(files)
        print(f"{split_file} -> {len(self.samples)} samples")

    def _filter(self, files):
        valid = []
        for base in files:
            paths = [
                os.path.join(self.rgb_dir, base+".JPG"),
                os.path.join(self.mask_dir, base+".png"),
                os.path.join(self.ms_dir, base+"_G.TIF"),
                os.path.join(self.ms_dir, base+"_R.TIF"),
                os.path.join(self.ms_dir, base+"_RE.TIF"),
                os.path.join(self.ms_dir, base+"_NIR.TIF"),
            ]
            if all(os.path.exists(p) for p in paths):
                valid.append(base)
        return valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        rgb = cv2.imread(os.path.join(self.rgb_dir, base+".JPG"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256,256))
        rgb = rgb.astype(np.float32)/255.0
        rgb = np.transpose(rgb, (2,0,1))

        bands = []
        for b in ["_G.TIF","_R.TIF","_RE.TIF","_NIR.TIF"]:
            img = cv2.imread(os.path.join(self.ms_dir, base+b), 0)
            img = cv2.resize(img, (256,256))
            img = img.astype(np.float32)/255.0
            bands.append(img)

        G, R, RE, NIR = bands

        if self.use_indices:
            eps = 1e-6
            NDVI = (NIR - R) / (NIR + R + eps)
            NDRE = (NIR - RE) / (NIR + RE + eps)
            NDVI = (NDVI + 1) / 2
            NDRE = (NDRE + 1) / 2
            ms = np.stack([G, R, RE, NIR, NDVI, NDRE], axis=0)
        else:
            ms = np.stack([G, R, RE, NIR], axis=0)

        mask = cv2.imread(os.path.join(self.mask_dir, base+".png"), 0)
        mask = cv2.resize(mask, (256,256))
        mask = (mask > 0).astype(np.int64)

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(ms, dtype=torch.float32),
            torch.tensor(mask)
        )

# -----------------------------
# CBAM
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) +
            self.fc(self.max_pool(x))
        )

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# -----------------------------
# Proposed Model
# -----------------------------
class LateFusionDeepLab(nn.Module):
    def __init__(self, use_ms=True, use_cbam=True, ms_channels=6):
        super().__init__()

        self.use_ms = use_ms
        self.use_cbam = use_cbam

        self.rgb_net = deeplabv3_resnet50(weights=None)

        if self.use_ms:
            self.ms_net = deeplabv3_resnet50(weights=None)
            self.ms_net.backbone.conv1 = nn.Conv2d(ms_channels,64,7,2,3,bias=False)

        in_channels = 256 if not use_ms else 512
        self.fusion_conv = nn.Conv2d(in_channels, 256, 1)

        if self.use_cbam:
            self.cbam = CBAM(256)

        self.classifier = nn.Conv2d(256, 2, 1)

    def forward(self, rgb, ms):
        input_size = rgb.shape[-2:]

        feat_rgb = self.rgb_net.backbone(rgb)['out']
        feat_rgb = self.rgb_net.classifier[0](feat_rgb)

        if self.use_ms:
            feat_ms = self.ms_net.backbone(ms)['out']
            feat_ms = self.ms_net.classifier[0](feat_ms)
            feat = torch.cat([feat_rgb, feat_ms], dim=1)
        else:
            feat_ms = torch.zeros_like(feat_rgb)
            feat = feat_rgb

        feat = self.fusion_conv(feat)

        if self.use_cbam:
            feat = self.cbam(feat)

        out = self.classifier(feat)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out, feat_rgb, feat_ms, feat

# -----------------------------
# SegFormer Model
# -----------------------------
# class SegFormerModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # self.model = SegformerForSemanticSegmentation.from_pretrained(
#         #     "nvidia/segformer-b0-finetuned-ade-512-512",
#         #     num_labels=2,
#         #     ignore_mismatched_sizes=True
#         # )
        

#     def forward(self, rgb):
#         outputs = self.model(pixel_values=rgb)
#         logits = outputs.logits
#         logits = F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
#         return logits

class SegFormerModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Create config (NO pretrained weights)
        config = SegformerConfig(
            num_labels=num_classes,
        )

        # Initialize model from scratch
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, rgb):
        outputs = self.model(pixel_values=rgb)
        logits = outputs.logits

        # Upsample to original image size
        logits = F.interpolate(
            logits,
            size=rgb.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return logits
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    root = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EXPERIMENTS = [
        # {"name": "DeepLab_RGB", "type": "deeplab", "use_ms": False, "use_cbam": False, "use_indices": False},
        {"name": "SegFormer", "type": "segformer"},
        # {"name": "PROPOSED", "type": "proposed", "use_ms": True, "use_cbam": True, "use_indices": True},
    ]

    NUM_RUNS = 5

    for exp in EXPERIMENTS:
        print(f"\n===== {exp['name']} =====")

        all_results = []

        for run in range(NUM_RUNS):
            print(f"\n--- Run {run+1} ---")
            set_seed(42 + run)

            train_loader = DataLoader(
                WeedyRiceDataset(root,"train_list.txt",exp.get("use_indices", True)),
                batch_size=2, shuffle=True
            )
            val_loader = DataLoader(
                WeedyRiceDataset(root,"val_list.txt",exp.get("use_indices", True)),
                batch_size=2
            )
            test_loader = DataLoader(
                WeedyRiceDataset(root,"test_list.txt",exp.get("use_indices", True)),
                batch_size=2
            )

            if exp["type"] == "segformer":
                model = SegFormerModel().to(device)

            elif exp["type"] == "deeplab":
                model = LateFusionDeepLab(use_ms=False, use_cbam=False, ms_channels=4).to(device)

            else:
                ms_channels = 6 if exp["use_indices"] else 4
                model = LateFusionDeepLab(
                    use_ms=exp["use_ms"],
                    use_cbam=exp["use_cbam"],
                    ms_channels=ms_channels
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            ce_loss = nn.CrossEntropyLoss()

            best_iou = 0
            best_path = f"{exp['name']}_run{run}.pth"

            for epoch in range(20):
                model.train()

                for rgb, ms, masks in train_loader:
                    rgb, ms, masks = rgb.to(device), ms.to(device), masks.to(device)

                    if exp["type"] == "segformer":
                        preds = model(rgb)
                    else:
                        preds, _, _, _ = model(rgb, ms)

                    loss = ce_loss(preds, masks)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val = evaluate(model, val_loader, device, exp["type"])
                print(f"Epoch {epoch+1} | IoU: {val['IoU']:.4f}")

                if val["IoU"] > best_iou:
                    best_iou = val["IoU"]
                    torch.save(model.state_dict(), best_path)

            model.load_state_dict(torch.load(best_path))
            test = evaluate(model, test_loader, device, exp["type"])

            print(f"Test: {test}")
            all_results.append(test)

        print(f"\n===== FINAL: {exp['name']} =====")
        for metric in all_results[0].keys():
            vals = [r[metric] for r in all_results]
            print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# -----------------------------
# SAMPLE PREDICTION DEMO
# -----------------------------
import matplotlib.pyplot as plt

print("\n🎨 Generating SegFormer Sample Predictions...\n")

# load trained model
model.load_state_dict(torch.load("SegFormer_run0.pth"))

model.eval()

# indices you want
sample_indices = [1, 5, 10, 15, 20]

fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 18))

with torch.no_grad():

    for row, idx in enumerate(sample_indices):

        # get sample
        rgb, ms, mask = test_loader.dataset[idx]

        # RGB visualization
        rgb_vis = rgb.permute(1, 2, 0).cpu().numpy()

        # input tensor
        rgb_input = rgb.unsqueeze(0).to(device)

        # prediction
        preds = model(rgb_input)

        # convert logits -> mask
        pred_mask = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()

        # ground truth
        gt_mask = mask.cpu().numpy()

        # -----------------------------
        # RGB IMAGE
        # -----------------------------
        axes[row, 0].imshow(rgb_vis)
        axes[row, 0].set_title(f"RGB Image (Index {idx})")
        axes[row, 0].axis("off")

        # -----------------------------
        # GROUND TRUTH
        # -----------------------------
        axes[row, 1].imshow(gt_mask, cmap='gray')
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis("off")

        # -----------------------------
        # PREDICTED MASK
        # -----------------------------
        axes[row, 2].imshow(pred_mask, cmap='gray')
        axes[row, 2].set_title("Predicted Mask")
        axes[row, 2].axis("off")

plt.tight_layout()

# optional save
plt.savefig(
    "segformer_predictions.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()