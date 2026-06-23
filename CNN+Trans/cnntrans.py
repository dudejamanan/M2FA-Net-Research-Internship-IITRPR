import os
import cv2
import torch
import timm
import random
import numpy as np
import csv

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import torchvision.models as models

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

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    prec = TP / (TP + FP + 1e-6)

    rec = TP / (TP + FN + 1e-6)

    iou = TP / (TP + FP + FN + 1e-6)

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    return acc, prec, rec, iou, dice

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):

    model.eval()

    total = {
        "Accuracy":0,
        "Precision":0,
        "Recall":0,
        "IoU":0,
        "Dice":0
    }

    count = 0

    with torch.no_grad():

        for rgb, masks in loader:

            rgb = rgb.to(device)

            masks = masks.to(device)

            preds = model(rgb)

            preds = F.interpolate(
                preds,
                size=masks.shape[1:],
                mode='bilinear',
                align_corners=False
            )

            acc, prec, rec, iou, dice = compute_metrics(preds, masks)

            for k, v in zip(
                total.keys(),
                [acc, prec, rec, iou, dice]
            ):
                total[k] += v

            count += 1

    return {k: v/count for k, v in total.items()}

# -----------------------------
# Dataset
# -----------------------------
class WeedyRiceDataset(Dataset):

    def __init__(self, root_dir, split_file):

        self.rgb_dir = os.path.join(root_dir, "RGB")

        self.mask_dir = os.path.join(root_dir, "Masks")

        with open(split_file, "r") as f:

            self.samples = [
                x.strip().replace(".JPG","").replace(".jpg","")
                for x in f
            ]

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        base = self.samples[idx]

        # -----------------------------
        # RGB IMAGE
        # -----------------------------
        rgb = cv2.imread(
            os.path.join(self.rgb_dir, base + ".JPG")
        )

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        rgb = cv2.resize(rgb, (224,224))

        rgb = rgb.astype(np.float32) / 255.0

        rgb = np.transpose(rgb, (2,0,1))

        # -----------------------------
        # MASK
        # -----------------------------
        mask = cv2.imread(
            os.path.join(self.mask_dir, base + ".png"),
            0
        )

        mask = cv2.resize(mask, (224,224))

        mask = (mask > 0).astype(np.int64)

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(mask)
        )

# -----------------------------
# CBAM
# -----------------------------
class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels, channels//8, 1),

            nn.ReLU(),

            nn.Conv2d(channels//8, channels, 1),

            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(2,1,kernel_size=7,padding=3),

            nn.Sigmoid()
        )

    def forward(self, x):

        # Channel Attention
        ca = self.channel_attention(x)

        x = x * ca

        # Spatial Attention
        avg = torch.mean(x, dim=1, keepdim=True)

        mx,_ = torch.max(x, dim=1, keepdim=True)

        sa = self.spatial_attention(
            torch.cat([avg,mx], dim=1)
        )

        x = x * sa

        return x

# -----------------------------
# CNN + Transformer Network
# -----------------------------
class CTFFNet(nn.Module):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # CNN Encoder
        # -----------------------------
        mobilenet = models.mobilenet_v2(
            pretrained=True
        )

        self.cnn_encoder = mobilenet.features

        # -----------------------------
        # Transformer Encoder
        # -----------------------------
        self.transformer = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True
        )

        # remove classifier
        self.transformer.head = nn.Identity()

        # -----------------------------
        # Transformer Projection
        # -----------------------------
        self.trans_proj = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU()
        )

        # -----------------------------
        # CNN Projection
        # -----------------------------
        self.cnn_proj = nn.Sequential(

            nn.Conv2d(1280, 256, 1),

            nn.BatchNorm2d(256),

            nn.ReLU()
        )

        # -----------------------------
        # CBAM Fusion
        # -----------------------------
        self.cbam = CBAM(512)

        # -----------------------------
        # Decoder
        # -----------------------------
        self.decoder = nn.Sequential(

    nn.ConvTranspose2d(
        512,
        256,
        kernel_size=2,
        stride=2
    ),

    nn.ReLU(),

    nn.ConvTranspose2d(
        256,
        128,
        kernel_size=2,
        stride=2
    ),

    nn.ReLU(),

    nn.ConvTranspose2d(
        128,
        64,
        kernel_size=2,
        stride=2
    ),

    nn.ReLU(),

    nn.ConvTranspose2d(
        64,
        32,
        kernel_size=2,
        stride=2
    ),

    nn.ReLU(),

    # NEW BLOCK
    nn.ConvTranspose2d(
        32,
        16,
        kernel_size=2,
        stride=2
    ),

    nn.ReLU()
    )
    # -----------------------------
    # Segmentation Head
    # -----------------------------
        self.seg_head = nn.Conv2d(
    in_channels=16,
    out_channels=2,
    kernel_size=1
    
)
    def forward(self, x):

        # -----------------------------
        # CNN Features
        # -----------------------------
        cnn_feat = self.cnn_encoder(x)

        cnn_feat = self.cnn_proj(cnn_feat)

        # shape:
        # [B,256,7,7]

        # -----------------------------
        # Transformer Features
        # -----------------------------
        trans_feat = self.transformer.forward_features(x)

        trans_feat = trans_feat[:,1:]

        trans_feat = self.trans_proj(trans_feat)

        B,N,C = trans_feat.shape

        trans_feat = trans_feat.permute(0,2,1)

        trans_feat = trans_feat.reshape(B,C,14,14)

        # resize
        trans_feat = F.interpolate(
            trans_feat,
            size=(7,7),
            mode='bilinear',
            align_corners=False
        )

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = torch.cat(
            [cnn_feat, trans_feat],
            dim=1
        )

        fused = self.cbam(fused)

        # -----------------------------
        # Decoder
        # -----------------------------
        out = self.decoder(fused)

        out = self.seg_head(out)

        return out

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    root = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    NUM_RUNS = 5

    all_results = []
    all_epoch_metrics = []
    for run in range(NUM_RUNS):
        epoch_metrics = []

        print(f"\n===== RUN {run+1} =====")

        set_seed(42 + run)

        train_loader = DataLoader(
            WeedyRiceDataset(root, "train_list.txt"),
            batch_size=8,
            shuffle=True
        )

        val_loader = DataLoader(
            WeedyRiceDataset(root, "val_list.txt"),
            batch_size=8
        )

        test_loader = DataLoader(
            WeedyRiceDataset(root, "test_list.txt"),
            batch_size=8
        )

        model = CTFFNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        ce_loss = nn.CrossEntropyLoss()

        best_iou = 0

        best_model = None

        # -----------------------------
        # TRAINING
        # -----------------------------
        for epoch in range(20):

            model.train()

            for rgb, masks in train_loader:

                rgb = rgb.to(device)

                masks = masks.to(device)

                preds = model(rgb)

                loss = ce_loss(preds, masks)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            val = evaluate(
                model,
                val_loader,
                device
            )

            print(
                f"Epoch {epoch+1} | "
                f"Val IoU: {val['IoU']:.4f}"
            )

            if val["IoU"] > best_iou:

                best_iou = val["IoU"]

                best_model = {
                    k:v.cpu().clone()
                    for k,v in model.state_dict().items()
                }

        # -----------------------------
        # TEST
        # -----------------------------
        model.load_state_dict(best_model)

        torch.save(
            model.state_dict(),
            "ctffnet_model.pth"
        )

        print("Model saved successfully!")

        test = evaluate(
            model,
            test_loader,
            device
        )

        print("Test:", test)

        all_results.append(test)
        df_run = pd.DataFrame(epoch_metrics)

        df_run.to_excel(
            f"run_{run+1}_epoch_metrics.xlsx",
            index=False
        )

    print(f"Saved run_{run+1}_epoch_metrics.xlsx")

    # -----------------------------
    # FINAL RESULTS
    # -----------------------------
    print("\n===== FINAL RESULTS =====")

    for metric in all_results[0].keys():

        val = evaluate(
            model,
            val_loader,
            device
        )

        record = {
        "Run": run + 1,
        "Epoch": epoch + 1,
        "Val_Accuracy": val["Accuracy"],
        "Val_Precision": val["Precision"],
        "Val_Recall": val["Recall"],
        "Val_IoU": val["IoU"],
        "Val_Dice": val["Dice"]
        }

        epoch_metrics.append(record)
        all_epoch_metrics.append(record)

        print(
            f"Epoch {epoch+1} | "
            f"Acc={val['Accuracy']:.4f} "
            f"Prec={val['Precision']:.4f} "
            f"Rec={val['Recall']:.4f} "
            f"IoU={val['IoU']:.4f} "
            f"Dice={val['Dice']:.4f}"
        )
df_all = pd.DataFrame(all_epoch_metrics)

df_all.to_excel(
    "all_runs_epoch_metrics.xlsx",
    index=False
)

print("Saved all_runs_epoch_metrics.xlsx")