from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# PATHS
# ============================================================

DATASET = Path(r"F:\MnM2\MnM2\dataset")
VAL_IMAGES = Path(r"F:\MnM2_Validation\imagesTs")
VAL_GT = Path(r"F:\MnM2_Validation\labelsVal")

OUT = Path(r"F:\MnM2_Validation\PPT_Images")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPER
# ============================================================

def load_nii(path):
    return nib.load(str(path)).get_fdata()


def normalize(img):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)

    if p99 <= p1:
        return np.zeros_like(img)

    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1)


# ============================================================
# SLIDE 3
# M&Ms-2 MRI + RV GROUND TRUTH
# ============================================================

print("\nCreating Slide 3 visual...")

# Use a validation patient with a good segmentation example
patient = "001"
phase = "ED"

# Original dataset path
img_path = DATASET / patient / f"{patient}_SA_{phase}.nii"
gt_path = DATASET / patient / f"{patient}_SA_{phase}_gt.nii"

if not img_path.exists():
    print("Original patient 001 not found. Using validation image.")

    img_path = VAL_IMAGES / f"{patient}_{phase}_0000.nii.gz"
    gt_path = VAL_GT / f"{patient}_{phase}.nii.gz"

img = load_nii(img_path)
gt = load_nii(gt_path)

# Convert original GT to RV only
# Original RV = label 3
if np.max(gt) > 1:
    gt = gt == 3
else:
    gt = gt > 0

# Find slice with largest RV
z = int(np.argmax(gt.sum(axis=(0, 1))))

img_slice = normalize(img[:, :, z])
gt_slice = gt[:, :, z]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
axes[0].set_title("Short-Axis Cardiac MRI", fontsize=15)
axes[0].axis("off")

axes[1].imshow(img_slice.T, cmap="gray", origin="lower")
axes[1].contour(
    gt_slice.T,
    levels=[0.5],
    linewidths=3
)
axes[1].set_title("RV Ground Truth", fontsize=15)
axes[1].axis("off")

fig.suptitle(
    "M&Ms-2 Dataset: Example of RV Annotation",
    fontsize=17
)

plt.tight_layout()

plt.savefig(
    OUT / "Slide_03_MnM2_MRI_RV_GT.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 3")


# ============================================================
# SLIDE 4
# ORIGINAL MULTI-CLASS MASK -> RV ONLY
# ============================================================

print("\nCreating Slide 4 visual...")

# Use same patient
img_path = DATASET / patient / f"{patient}_SA_{phase}.nii"
gt_path = DATASET / patient / f"{patient}_SA_{phase}_gt.nii"

if not img_path.exists():
    img_path = VAL_IMAGES / f"{patient}_{phase}_0000.nii.gz"
    gt_path = VAL_GT / f"{patient}_{phase}.nii.gz"

img = load_nii(img_path)
original_gt = load_nii(gt_path)

# If validation GT is already binary
if np.max(original_gt) <= 1:
    # Use original dataset if available
    original_gt_path = DATASET / patient / f"{patient}_SA_{phase}_gt.nii"
    if original_gt_path.exists():
        original_gt = load_nii(original_gt_path)

z = int(np.argmax((original_gt == 3).sum(axis=(0, 1))))

image_slice = normalize(img[:, :, z])
original_slice = original_gt[:, :, z]

# RV only
rv_only = original_slice == 3

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
axes[0].set_title("MRI", fontsize=15)
axes[0].axis("off")

axes[1].imshow(original_slice.T, cmap="viridis", origin="lower")
axes[1].set_title(
    "Original Multi-Class Annotation\n"
    "0 Background | 1 LV | 2 Myocardium | 3 RV",
    fontsize=12
)
axes[1].axis("off")

axes[2].imshow(image_slice.T, cmap="gray", origin="lower")
axes[2].contour(
    rv_only.T,
    levels=[0.5],
    linewidths=3
)
axes[2].set_title("RV-Only Target Mask", fontsize=15)
axes[2].axis("off")

fig.suptitle(
    "Conversion to Binary RV Segmentation",
    fontsize=17
)

plt.tight_layout()

plt.savefig(
    OUT / "Slide_04_RV_Label_Conversion.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 4")


# ============================================================
# SLIDE 5
# PATIENT LEVEL SPLIT
# ============================================================

print("\nCreating Slide 5 visual...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

# Title
ax.text(
    0.5, 0.92,
    "Patient-Level Data Split",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

# Main total
ax.text(
    0.5, 0.76,
    "360 Patients",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

# Boxes
boxes = [
    (0.20, "Training\n251 patients\n70%"),
    (0.50, "Validation\n55 patients\n15%"),
    (0.80, "Test\n54 patients\n15%")
]

for x, text in boxes:

    box = FancyBboxPatch(
        (x - 0.12, 0.38),
        0.24,
        0.20,
        boxstyle="round,pad=0.02",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x,
        0.48,
        text,
        ha="center",
        va="center",
        fontsize=15
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.5, 0.71),
            (x, 0.59),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8
        )
    )

# ED/ES explanation
ax.text(
    0.5,
    0.20,
    "Each patient's ED and ES images remain in the same partition",
    ha="center",
    va="center",
    fontsize=16
)

ax.text(
    0.5,
    0.11,
    "→ Prevents patient-level data leakage",
    ha="center",
    va="center",
    fontsize=15
)

plt.savefig(
    OUT / "Slide_05_Patient_Level_Split.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 5")


# ============================================================
# SLIDE 6
# NNUNET PIPELINE
# ============================================================

print("\nCreating Slide 6 visual...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

ax.text(
    0.5,
    0.92,
    "3D nnU-Net Baseline Pipeline",
    ha="center",
    fontsize=22,
    fontweight="bold"
)

steps = [
    ("Cardiac MRI\nSA ED/ES", 0.12),
    ("Preprocessing\n& Resampling", 0.34),
    ("3D nnU-Net\nFull Resolution", 0.58),
    ("RV\nPrediction", 0.84)
]

for i, (text, x) in enumerate(steps):

    box = FancyBboxPatch(
        (x - 0.10, 0.38),
        0.20,
        0.25,
        boxstyle="round,pad=0.03",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x,
        0.505,
        text,
        ha="center",
        va="center",
        fontsize=14
    )

    if i < len(steps) - 1:

        ax.add_patch(
            FancyArrowPatch(
                (x + 0.10, 0.505),
                (steps[i + 1][1] - 0.10, 0.505),
                arrowstyle="->",
                mutation_scale=20,
                linewidth=2
            )
        )

ax.text(
    0.5,
    0.18,
    "Model: nnU-Net v2 | Configuration: 3D Full Resolution | Fold 0",
    ha="center",
    fontsize=14
)

plt.savefig(
    OUT / "Slide_06_nnUNet_Pipeline.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 6")


# ============================================================
# SLIDE 8
# VALIDATION WORKFLOW
# ============================================================

print("\nCreating Slide 8 visual...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

ax.text(
    0.5,
    0.92,
    "Independent Patient-Level Validation",
    ha="center",
    fontsize=22,
    fontweight="bold"
)

steps = [
    ("55 Validation\nPatients", 0.12),
    ("ED + ES\n110 Cases", 0.34),
    ("nnU-Net\nPrediction", 0.56),
    ("Compare with\nGround Truth", 0.76),
    ("Dice +\nHD95", 0.92)
]

for i, (text, x) in enumerate(steps):

    width = 0.16 if i < 4 else 0.13

    box = FancyBboxPatch(
        (x - width / 2, 0.38),
        width,
        0.25,
        boxstyle="round,pad=0.02",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x,
        0.505,
        text,
        ha="center",
        va="center",
        fontsize=13
    )

    if i < len(steps) - 1:

        next_x = steps[i + 1][1]

        ax.add_patch(
            FancyArrowPatch(
                (x + width / 2, 0.505),
                (next_x - (0.16 if i + 1 < 4 else 0.13) / 2, 0.505),
                arrowstyle="->",
                mutation_scale=18,
                linewidth=1.8
            )
        )

plt.savefig(
    OUT / "Slide_08_Validation_Workflow.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 8")


# ============================================================
# SLIDE 12
# QUALITATIVE ANALYSIS
# ============================================================

print("\nCreating Slide 12 visual...")

# Use one of the difficult cases as an example
patient = "261"
phase = "ES"

img_path = VAL_IMAGES / f"{patient}_{phase}_0000.nii.gz"
gt_path = VAL_GT / f"{patient}_{phase}.nii.gz"

# Prediction
pred_path = Path(
    r"F:\MnM2_Validation\predictions"
) / f"{patient}_{phase}.nii.gz"

img = load_nii(img_path)
gt = load_nii(gt_path) > 0
pred = load_nii(pred_path) > 0

z = int(np.argmax(gt.sum(axis=(0, 1))))

image_slice = normalize(img[:, :, z])
gt_slice = gt[:, :, z]
pred_slice = pred[:, :, z]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
axes[0].set_title("MRI", fontsize=15)
axes[0].axis("off")

axes[1].imshow(image_slice.T, cmap="gray", origin="lower")
axes[1].contour(
    gt_slice.T,
    levels=[0.5],
    linewidths=3
)
axes[1].set_title("Ground Truth", fontsize=15)
axes[1].axis("off")

axes[2].imshow(image_slice.T, cmap="gray", origin="lower")
axes[2].contour(
    gt_slice.T,
    levels=[0.5],
    linewidths=2
)
axes[2].contour(
    pred_slice.T,
    levels=[0.5],
    linewidths=2,
    linestyles="--"
)
axes[2].set_title("GT vs Prediction", fontsize=15)
axes[2].axis("off")

fig.suptitle(
    "Qualitative Failure Analysis Example — Patient 261 ES",
    fontsize=17
)

plt.tight_layout()

plt.savefig(
    OUT / "Slide_12_Qualitative_Analysis.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 12")


# ============================================================
# SLIDE 18
# CURRENT FINDINGS
# ============================================================

print("\nCreating Slide 18 visual...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

ax.text(
    0.5,
    0.90,
    "Current Findings",
    ha="center",
    fontsize=24,
    fontweight="bold"
)

findings = [
    "Mean Dice: 0.9223",
    "ED Dice > ES Dice",
    "Most validation cases show good RV overlap",
    "A small number of difficult cases cause larger errors",
    "Qualitative failure analysis is needed"
]

y = 0.72

for text in findings:

    box = FancyBboxPatch(
        (0.18, y - 0.04),
        0.64,
        0.09,
        boxstyle="round,pad=0.02",
        linewidth=1.8
    )

    ax.add_patch(box)

    ax.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=16
    )

    y -= 0.13

plt.savefig(
    OUT / "Slide_18_Current_Findings.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 18")


# ============================================================
# SLIDE 19
# NEXT STEPS
# ============================================================

print("\nCreating Slide 19 visual...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

ax.text(
    0.5,
    0.92,
    "Next Steps",
    ha="center",
    fontsize=24,
    fontweight="bold"
)

steps = [
    ("Qualitative\nFailure Analysis", 0.12),
    ("RV EDV / ESV\n& EF", 0.34),
    ("Centre / Disease /\nView Analysis", 0.56),
    ("Finalize Model\n& Protocol", 0.76),
    ("Final 54-Patient\nTest Evaluation", 0.92)
]

for i, (text, x) in enumerate(steps):

    width = 0.16 if i < 4 else 0.13

    box = FancyBboxPatch(
        (x - width / 2, 0.40),
        width,
        0.25,
        boxstyle="round,pad=0.02",
        linewidth=2
    )

    ax.add_patch(box)

    ax.text(
        x,
        0.525,
        text,
        ha="center",
        va="center",
        fontsize=12
    )

    if i < len(steps) - 1:

        next_x = steps[i + 1][1]

        ax.add_patch(
            FancyArrowPatch(
                (x + width / 2, 0.525),
                (next_x - (0.16 if i + 1 < 4 else 0.13) / 2, 0.525),
                arrowstyle="->",
                mutation_scale=18,
                linewidth=1.8
            )
        )

plt.savefig(
    OUT / "Slide_19_Next_Steps.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("Saved Slide 19")


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("ALL ADDITIONAL PPT VISUALS CREATED")
print("======================================")
print(OUT)