import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random

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
def evaluate(model, loader, device):
    model.eval()

    total_acc = total_prec = total_rec = total_iou = total_dice = 0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)

            acc, prec, rec, iou, dice = compute_metrics(preds, masks)

            total_acc += acc
            total_prec += prec
            total_rec += rec
            total_iou += iou
            total_dice += dice
            count += 1

    return {
        "Accuracy": total_acc / count,
        "Precision": total_prec / count,
        "Recall": total_rec / count,
        "IoU": total_iou / count,
        "Dice": total_dice / count
    }

# -----------------------------
# DATASET (RGB + MS)
# -----------------------------
class WeedyRiceDataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.ms_dir = os.path.join(root_dir, "Multispectral")
        self.mask_dir = os.path.join(root_dir, "Masks")

        with open(split_file, "r") as f:
            raw_list = [line.strip() for line in f.readlines()]

        self.samples = self._filter_valid_samples(raw_list)
        print(f"{split_file} -> {len(self.samples)} valid samples loaded")

    def _filter_valid_samples(self, file_list):
        valid = []
        for name in file_list:
            base = name.replace(".JPG", "").replace(".jpg", "")

            paths = [
                os.path.join(self.rgb_dir, base + ".JPG"),
                os.path.join(self.mask_dir, base + ".png"),
                os.path.join(self.ms_dir, base + "_G.TIF"),
                os.path.join(self.ms_dir, base + "_R.TIF"),
                os.path.join(self.ms_dir, base + "_RE.TIF"),
                os.path.join(self.ms_dir, base + "_NIR.TIF"),
            ]

            if all(os.path.exists(p) for p in paths):
                valid.append(base)

        return valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        # -------- RGB --------
        rgb = cv2.imread(os.path.join(self.rgb_dir, base + ".JPG"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256))
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # (3,H,W)

        # -------- MULTISPECTRAL --------
        ms_bands = []
        for suffix in ["_G.TIF", "_R.TIF", "_RE.TIF", "_NIR.TIF"]:
            img = cv2.imread(os.path.join(self.ms_dir, base + suffix), 0)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            ms_bands.append(img)

        ms = np.stack(ms_bands, axis=0)  # (4,H,W)

        # -------- FUSION --------
        fused = np.concatenate([rgb, ms], axis=0)  # (7,H,W)

        # -------- MASK --------
        mask = cv2.imread(os.path.join(self.mask_dir, base + ".png"), 0)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.int64)

        return torch.tensor(fused, dtype=torch.float32), torch.tensor(mask)

# -----------------------------
# PATCH EMBEDDING
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=7, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        return x, (H, W)

# -----------------------------
# TRANSFORMER MODEL
# -----------------------------
class SimpleTransformerSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.patch_embed = PatchEmbed()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x, (H_p, W_p) = self.patch_embed(x)

        tokens = x.flatten(2).transpose(1, 2)

        tokens = self.transformer(tokens)

        x = tokens.transpose(1, 2).reshape(B, 128, H_p, W_p)

        x = self.decoder(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x

# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(pred, target):
    pred = torch.softmax(pred, dim=1)[:, 1]
    target = target.float()

    intersection = (pred * target).sum()
    return 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    root = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"

    NUM_RUNS = 5
    all_results = []

    for run in range(NUM_RUNS):

        print(f"\n================ RUN {run+1}/{NUM_RUNS} ================\n")

        set_seed(42 + run)

        train_loader = DataLoader(WeedyRiceDataset(root, "train_list.txt"), batch_size=2, shuffle=True)
        val_loader   = DataLoader(WeedyRiceDataset(root, "val_list.txt"), batch_size=2)
        test_loader  = DataLoader(WeedyRiceDataset(root, "test_list.txt"), batch_size=2)

        model = SimpleTransformerSeg()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(device))

        best_iou = 0

        print("🚀 Training started...")

        for epoch in range(20):
            model.train()

            total_loss = 0
            total_iou = 0
            count = 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)

                loss = ce_loss(preds, masks) + dice_loss(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, _, _, iou, _ = compute_metrics(preds, masks)
                total_iou += iou
                count += 1

            val_metrics = evaluate(model, val_loader, device)

            print(f"""
Epoch {epoch+1}
Train Loss: {total_loss:.4f}
Train IoU: {total_iou/count:.4f}

--- Validation ---
IoU: {val_metrics['IoU']:.4f}
Dice: {val_metrics['Dice']:.4f}
""")

            if val_metrics["IoU"] > best_iou:
                best_iou = val_metrics["IoU"]
                torch.save(model.state_dict(), f"best_model_run{run}.pth")

        print("\n🧪 Testing best model...")

        model.load_state_dict(torch.load(f"best_model_run{run}.pth"))
        test_metrics = evaluate(model, test_loader, device)

        print(f"""
===== RUN {run+1} TEST RESULTS =====
Accuracy: {test_metrics["Accuracy"]:.4f}
Precision: {test_metrics["Precision"]:.4f}
Recall: {test_metrics["Recall"]:.4f}
IoU: {test_metrics["IoU"]:.4f}
Dice: {test_metrics["Dice"]:.4f}
""")

        all_results.append(test_metrics)

# -----------------------------
# SAMPLE PREDICTION DEMO
# -----------------------------
import matplotlib.pyplot as plt

print("\n🎨 Generating Sample Predictions...\n")

model.eval()

# indices you want
sample_indices = [1, 5, 10, 15, 20]

fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 16))

with torch.no_grad():

    for row, idx in enumerate(sample_indices):

        # get sample
        image, mask = test_loader.dataset[idx]

        # keep RGB only for visualization
        rgb = image[:3].permute(1, 2, 0).cpu().numpy()

        # input to model
        input_tensor = image.unsqueeze(0).to(device)

        # prediction
        pred = model(input_tensor)

        # convert logits -> class map
        pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        # ground truth
        gt_mask = mask.cpu().numpy()

        # -----------------------------
        # SHOW RGB IMAGE
        # -----------------------------
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"RGB Image (Index {idx})")
        axes[row, 0].axis("off")

        # -----------------------------
        # SHOW GROUND TRUTH
        # -----------------------------
        axes[row, 1].imshow(gt_mask, cmap='gray')
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis("off")

        # -----------------------------
        # SHOW PREDICTION
        # -----------------------------
        axes[row, 2].imshow(pred_mask, cmap='gray')
        axes[row, 2].set_title("Predicted Mask")
        axes[row, 2].axis("off")

plt.tight_layout()
plt.show()

print("\n================ FINAL AVERAGED RESULTS ================\n")

for key in all_results[0]:
    values = [res[key] for res in all_results]
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")