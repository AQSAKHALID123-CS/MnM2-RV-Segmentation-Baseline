from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

IMG_DIR = Path(r"F:\MnM2_Validation\imagesTs")
GT_DIR = Path(r"F:\MnM2_Validation\labelsVal")
PRED_DIR = Path(r"F:\MnM2_Validation\predictions")

METRICS_FILE = Path(r"F:\MnM2_Validation\validation_metrics.csv")
PATIENT_FILE = Path(r"F:\MnM2_Validation\patient_level_results.csv")

OUT_DIR = Path(r"F:\MnM2_Validation\PPT_Images")
OVERLAY_DIR = OUT_DIR / "Worst5_Overlays"
CHART_DIR = OUT_DIR / "Charts"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Worst 5 patients
WORST_PATIENTS = ["261", "349", "186", "084", "233"]
PHASES = ["ED", "ES"]


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_nifti(path):
    return nib.load(str(path)).get_fdata()


def get_middle_gt_slice(gt):
    """
    Select slice having maximum RV area.
    """
    area = (gt > 0).sum(axis=(0, 1))
    return int(np.argmax(area))


def normalize_image(img):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)

    if p99 <= p1:
        return np.zeros_like(img)

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1)

    return img


# ============================================================
# 1. WORST-5 OVERLAYS
# ============================================================

print("\nCreating worst-5 overlay images...")

for patient in WORST_PATIENTS:

    for phase in PHASES:

        name = f"{patient}_{phase}"

        img_path = IMG_DIR / f"{name}_0000.nii.gz"
        gt_path = GT_DIR / f"{name}.nii.gz"
        pred_path = PRED_DIR / f"{name}.nii.gz"

        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        if not gt_path.exists():
            print("Missing GT:", gt_path)
            continue

        if not pred_path.exists():
            print("Missing prediction:", pred_path)
            continue

        img = load_nifti(img_path)
        gt = load_nifti(gt_path)
        pred = load_nifti(pred_path)

        # Convert to binary
        gt = gt > 0
        pred = pred > 0

        # Select slice with largest GT RV area
        z = get_middle_gt_slice(gt)

        image_slice = normalize_image(img[:, :, z])
        gt_slice = gt[:, :, z]
        pred_slice = pred[:, :, z]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # ----------------------------------------------------
        # MRI
        # ----------------------------------------------------
        axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("MRI")
        axes[0].axis("off")

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------
        axes[1].imshow(image_slice.T, cmap="gray", origin="lower")
        axes[1].contour(gt_slice.T, levels=[0.5], linewidths=2)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        axes[2].imshow(image_slice.T, cmap="gray", origin="lower")
        axes[2].contour(pred_slice.T, levels=[0.5], linewidths=2)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------
        axes[3].imshow(image_slice.T, cmap="gray", origin="lower")

        axes[3].contour(
            gt_slice.T,
            levels=[0.5],
            linewidths=2,
            linestyles="-",
            label="Ground Truth"
        )

        axes[3].contour(
            pred_slice.T,
            levels=[0.5],
            linewidths=2,
            linestyles="--",
            label="Prediction"
        )

        axes[3].set_title("GT vs Prediction")
        axes[3].axis("off")

        fig.suptitle(
            f"Patient {patient} - {phase}",
            fontsize=16
        )

        plt.tight_layout()

        output = OVERLAY_DIR / f"{patient}_{phase}_overlay.png"

        plt.savefig(
            output,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output)


# ============================================================
# 2. OVERALL METRICS CHART
# ============================================================

print("\nCreating overall metrics chart...")

df = pd.read_csv(METRICS_FILE)

mean_dice = df["Dice"].mean()
median_dice = df["Dice"].median()
mean_hd95 = df["HD95_mm"].mean()
median_hd95 = df["HD95_mm"].median()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(
    ["Mean Dice", "Median Dice"],
    [mean_dice, median_dice]
)

axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Dice")
axes[0].set_title("Overall Dice")

axes[1].bar(
    ["Mean HD95", "Median HD95"],
    [mean_hd95, median_hd95]
)

axes[1].set_ylabel("HD95 (mm)")
axes[1].set_title("Overall HD95")

plt.tight_layout()

output = CHART_DIR / "overall_metrics.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", output)


# ============================================================
# 3. ED VS ES DICE
# ============================================================

print("\nCreating ED vs ES chart...")

# Phase is already available in the CSV
ed_dice = df[df["Phase"] == "ED"]["Dice"].mean()
es_dice = df[df["Phase"] == "ES"]["Dice"].mean()

ed_hd95 = df[df["Phase"] == "ED"]["HD95_mm"].mean()
es_hd95 = df[df["Phase"] == "ES"]["HD95_mm"].mean()

print("ED Dice:", ed_dice)
print("ES Dice:", es_dice)
print("ED HD95:", ed_hd95)
print("ES HD95:", es_hd95)

print("ED Dice:", ed_dice)
print("ES Dice:", es_dice)
print("ED HD95:", ed_hd95)
print("ES HD95:", es_hd95)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(
    ["ED", "ES"],
    [ed_dice, es_dice]
)

axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Mean Dice")
axes[0].set_title("ED vs ES - Dice")

axes[1].bar(
    ["ED", "ES"],
    [ed_hd95, es_hd95]
)

axes[1].set_ylabel("Mean HD95 (mm)")
axes[1].set_title("ED vs ES - HD95")

plt.tight_layout()

output = CHART_DIR / "ED_vs_ES.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", output)


# ============================================================
# 4. WORST 5 PATIENTS CHART
# ============================================================

print("\nCreating worst-5 chart...")

patient_df = pd.read_csv(PATIENT_FILE)

patient_df["Patient"] = (
    patient_df["Patient"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(3)
)

worst_df = (
    patient_df[
        patient_df["Patient"].isin(WORST_PATIENTS)
    ]
    .copy()
)

worst_df = worst_df.sort_values("Mean_Dice")

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    worst_df["Patient"],
    worst_df["Mean_Dice"]
)

ax.set_ylim(0, 1)
ax.set_xlabel("Patient")
ax.set_ylabel("Mean Dice")
ax.set_title("Five Poorest-Performing Patients")

plt.tight_layout()

output = CHART_DIR / "worst5_patients.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", output)


print("\n======================================")
print("ALL PPT IMAGES CREATED")
print("======================================")
print("Output folder:")
print(OUT_DIR)
print("\nOverlays:")
print(OVERLAY_DIR)
print("\nCharts:")
print(CHART_DIR)