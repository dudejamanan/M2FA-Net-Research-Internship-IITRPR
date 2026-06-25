import os
import cv2
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

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
# -----------------------------
# Metrics
# -----------------------------
def compute_confusion(preds, masks):

    preds = torch.argmax(preds, dim=1)

    preds = preds.view(-1)
    masks = masks.view(-1)

    TP = ((preds == 1) & (masks == 1)).sum().item()
    FP = ((preds == 1) & (masks == 0)).sum().item()
    FN = ((preds == 0) & (masks == 1)).sum().item()
    TN = ((preds == 0) & (masks == 0)).sum().item()

    return TP, FP, FN, TN


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):

    model.eval()

    TP_total = 0
    FP_total = 0
    FN_total = 0
    TN_total = 0

    with torch.no_grad():

        for rgb, ms, masks in loader:

            rgb = rgb.to(device)
            ms = ms.to(device)
            masks = masks.to(device)

            preds = model(rgb, ms)

            preds = F.interpolate(
                preds,
                size=masks.shape[1:],
                mode='bilinear',
                align_corners=False
            )

            TP, FP, FN, TN = compute_confusion(preds, masks)

            TP_total += TP
            FP_total += FP
            FN_total += FN
            TN_total += TN

    accuracy = (
        TP_total + TN_total
    ) / (
        TP_total + TN_total + FP_total + FN_total + 1e-6
    )

    precision = TP_total / (
        TP_total + FP_total + 1e-6
    )

    recall = TP_total / (
        TP_total + FN_total + 1e-6
    )

    iou = TP_total / (
        TP_total + FP_total + FN_total + 1e-6
    )

    dice = (2 * TP_total) / (
        2 * TP_total + FP_total + FN_total + 1e-6
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "Dice": dice
    }

# -----------------------------
# Dataset
# -----------------------------
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

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        base = self.samples[idx]

        # -----------------------------
        # RGB
        # -----------------------------
        rgb = cv2.imread(
            os.path.join(self.rgb_dir, base + ".JPG")
        )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, (224,224))

        rgb = rgb.astype(np.float32) / 255.0

        rgb = np.transpose(rgb, (2,0,1))

        # -----------------------------
        # Multispectral
        # -----------------------------
        bands = []

        for b in ["_G.TIF","_R.TIF","_RE.TIF","_NIR.TIF"]:

            img = cv2.imread(
                os.path.join(self.ms_dir, base + b),
                0
            )

            img = cv2.resize(img, (224,224))

            img = img.astype(np.float32) / 255.0

            bands.append(img)

        ms = np.stack(bands, axis=0)

        # -----------------------------
        # Mask
        # -----------------------------
        mask = cv2.imread(
            os.path.join(self.mask_dir, base + ".png"),
            0
        )

        mask = cv2.resize(
        mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
        )

        mask = (mask > 0).astype(np.int64)

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(ms, dtype=torch.float32),
            torch.tensor(mask)
        )

# -----------------------------
# Swin + U-Net
# -----------------------------
class SwinUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # RGB Branch
        self.rgb_net = smp.Unet(
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        # Multispectral Branch
        self.ms_net = smp.Unet(
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights=None,
            in_channels=4,
            classes=2
        )

        # IMPORTANT FOR 256x256
        self.rgb_net.encoder.model.patch_embed.strict_img_size = False
        self.ms_net.encoder.model.patch_embed.strict_img_size = False

        # Fusion Layer
        self.fusion = nn.Sequential(

            nn.Conv2d(2, 16, kernel_size=3, padding=1),

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),

            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, rgb, ms):

        out_rgb = self.rgb_net(rgb)

        out_ms = self.ms_net(ms)

        out = out_rgb + out_ms

        out = self.fusion(out)

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

    for run in range(NUM_RUNS):

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

        model = SwinUNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        ce_loss = nn.CrossEntropyLoss()

        best_iou = 0

        best_model = None

        # -----------------------------
        # Training
        # -----------------------------
        for epoch in range(20):

            model.train()

            for rgb, ms, masks in train_loader:

                rgb = rgb.to(device)

                ms = ms.to(device)

                masks = masks.to(device)

                preds = model(rgb, ms)

                loss = ce_loss(preds, masks)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            val = evaluate(model, val_loader, device)

            print(
                f"Epoch {epoch+1} | "
                f"Val IoU: {val['IoU']:.4f}"
            )

            if val["IoU"] > best_iou:

                best_iou = val["IoU"]

                torch.save(
                model.state_dict(),
                "best_swin_unet.pth"
                )

        # -----------------------------
        # TEST
        # -----------------------------
        model.load_state_dict(
        torch.load(
            "best_swin_unet.pth",
            map_location=device
            )
        )

        model.to(device)

        test = evaluate(model, test_loader, device)

        print("Test:", test)

        all_results.append(test)

    # -----------------------------
    # FINAL RESULTS
    # -----------------------------
    print("\n===== FINAL RESULTS =====")

    for metric in all_results[0].keys():

        vals = [r[metric] for r in all_results]

        print(
            f"{metric}: "
            f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        )
    model.load_state_dict(
    torch.load(
        "best_swin_unet.pth",
        map_location=device
    )
)

    # -----------------------------
    # SAVE CSV
    # -----------------------------
    with open(
        "swin_unet_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Run",
            "Accuracy",
            "Precision",
            "Recall",
            "IoU",
            "Dice"
        ])

        for i, r in enumerate(all_results):

            writer.writerow([
                i+1,
                r["Accuracy"],
                r["Precision"],
                r["Recall"],
                r["IoU"],
                r["Dice"]
            ])

# -----------------------------
# Visualization
# -----------------------------
import matplotlib.pyplot as plt

model.eval()

indices = [1, 5, 10, 15, 20]

fig, axes = plt.subplots(
    len(indices),
    3,
    figsize=(12, 4 * len(indices))
)

with torch.no_grad():

    for i, idx in enumerate(indices):

        rgb, ms, mask = test_loader.dataset[idx]

        rgb_input = rgb.unsqueeze(0).to(device)

        ms_input = ms.unsqueeze(0).to(device)

        pred = model(rgb_input, ms_input)

        pred = torch.argmax(
            pred,
            dim=1
        ).squeeze(0).cpu().numpy()

        rgb_img = rgb.permute(1,2,0).cpu().numpy()

        # -----------------------------
        # INPUT IMAGE
        # -----------------------------
        axes[i,0].imshow(rgb_img)

        axes[i,0].set_title(f"Input {idx}")

        axes[i,0].axis("off")

        # -----------------------------
        # GROUND TRUTH
        # -----------------------------
        axes[i,1].imshow(
            mask.cpu().numpy(),
            cmap='gray'
        )

        axes[i,1].set_title(
            f"Ground Truth {idx}"
        )

        axes[i,1].axis("off")

        # -----------------------------
        # PREDICTION
        # -----------------------------
        overlay = rgb_img.copy()

        overlay[pred == 1] = [1, 0, 0]

        axes[i,2].imshow(overlay)

        axes[i,2].set_title(
            f"Prediction {idx}"
        )

        axes[i,2].axis("off")

plt.tight_layout()

plt.savefig("all_predictions.png")

plt.show()