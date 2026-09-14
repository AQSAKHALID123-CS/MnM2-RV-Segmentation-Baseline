from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt


PRED_DIR = Path(r"F:\MnM2_Validation\predictions")
GT_DIR = Path(r"F:\MnM2_Validation\labelsVal")
OUTPUT_FILE = Path(r"F:\MnM2_Validation\validation_metrics.csv")


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    return 2.0 * intersection / (pred.sum() + gt.sum())


def hd95(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)

    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)

    distances_pred_to_gt = dt_gt[pred_surface]
    distances_gt_to_pred = dt_pred[gt_surface]

    all_distances = np.concatenate([
        distances_pred_to_gt,
        distances_gt_to_pred
    ])

    return np.percentile(all_distances, 95)


results = []

gt_files = sorted(GT_DIR.glob("*.nii.gz"))

print("GT cases:", len(gt_files))

for gt_file in gt_files:

    case_id = gt_file.stem.replace(".nii", "")
    pred_file = PRED_DIR / f"{case_id}.nii.gz"

    if not pred_file.exists():
        print("Missing prediction:", case_id)
        continue

    gt_img = nib.load(str(gt_file))
    pred_img = nib.load(str(pred_file))

    gt = gt_img.get_fdata() > 0
    pred = pred_img.get_fdata() > 0

    spacing = gt_img.header.get_zooms()[:3]

    dice = dice_score(pred, gt)
    hd = hd95(pred, gt, spacing)

    patient = case_id.split("_")[0]
    phase = case_id.split("_")[1]

    results.append([
        patient,
        phase,
        dice,
        hd
    ])

    print(
        f"{case_id}: "
        f"Dice={dice:.4f}, "
        f"HD95={hd:.2f} mm"
    )


# Save CSV
import csv

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Patient",
        "Phase",
        "Dice",
        "HD95_mm"
    ])

    writer.writerows(results)


dice_values = np.array([r[2] for r in results], dtype=float)
hd_values = np.array([
    r[3] for r in results
    if not np.isnan(r[3])
], dtype=float)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

print("Cases evaluated:", len(results))

print("\nDice:")
print("Mean   :", np.mean(dice_values))
print("Median :", np.median(dice_values))
print("Std    :", np.std(dice_values))
print("Min    :", np.min(dice_values))
print("Max    :", np.max(dice_values))

print("\nHD95 (mm):")
print("Mean   :", np.mean(hd_values))
print("Median :", np.median(hd_values))
print("Std    :", np.std(hd_values))
print("Min    :", np.min(hd_values))
print("Max    :", np.max(hd_values))

print("\nSaved to:")
print(OUTPUT_FILE)