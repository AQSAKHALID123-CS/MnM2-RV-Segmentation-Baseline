import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
IMG_DIR = Path(r"F:\MnM2_Validation\imagesTs")
GT_DIR = Path(r"F:\MnM2_Validation\labelsVal")
PRED_DIR = Path(r"F:\MnM2_Validation\predictions")
OUT_DIR = Path(r"F:\MnM2_Validation\overlays")

OUT_DIR.mkdir(exist_ok=True)

# 5 worst patients
patients = ["261", "349", "186", "84", "233"]
phases = ["ED", "ES"]


def load_nii(path):
    return nib.load(str(path)).get_fdata()


for patient in patients:

    for phase in phases:

        img_path = IMG_DIR / f"{patient}_{phase}_0000.nii.gz"
        gt_path = GT_DIR / f"{patient}_{phase}.nii.gz"
        pred_path = PRED_DIR / f"{patient}_{phase}.nii.gz"

        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        if not gt_path.exists():
            print("Missing GT:", gt_path)
            continue

        if not pred_path.exists():
            print("Missing prediction:", pred_path)
            continue

        image = load_nii(img_path)
        gt = load_nii(gt_path)
        pred = load_nii(pred_path)

        gt = gt > 0
        pred = pred > 0

        # Find slice with maximum GT RV area
        slice_scores = gt.sum(axis=(0, 1))
        z = int(np.argmax(slice_scores))

        img_slice = image[:, :, z]
        gt_slice = gt[:, :, z]
        pred_slice = pred[:, :, z]

        # Normalize image for display
        p1, p99 = np.percentile(img_slice, [1, 99])

        if p99 > p1:
            img_display = np.clip(
                (img_slice - p1) / (p99 - p1),
                0,
                1
            )
        else:
            img_display = img_slice

        # -----------------------------
        # Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(
            img_display.T,
            cmap="gray",
            origin="lower"
        )

        # Ground Truth contour
        if gt_slice.any():
            ax.contour(
                gt_slice.T,
                levels=[0.5],
                linewidths=2,
                colors="lime"
            )

        # Prediction contour
        if pred_slice.any():
            ax.contour(
                pred_slice.T,
                levels=[0.5],
                linewidths=2,
                colors="red"
            )

        ax.set_title(
            f"Patient {patient} | {phase} | Slice {z}\n"
            "GT = Green | Prediction = Red"
        )

        ax.axis("off")

        output_path = OUT_DIR / f"{patient}_{phase}.png"

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        print("Saved:", output_path)


print("\n==============================")
print("OVERLAYS COMPLETE")
print("==============================")
print("Output folder:")
print(OUT_DIR)