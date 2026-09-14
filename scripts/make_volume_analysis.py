from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

VAL_IMAGES = Path(r"F:\MnM2_Validation\imagesTs")
VAL_GT = Path(r"F:\MnM2_Validation\labelsVal")
PRED = Path(r"F:\MnM2_Validation\predictions")

OUT = Path(r"F:\MnM2_Validation\Volume_Analysis")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def load_nii(path):
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.header.get_zooms()[:3]


def calculate_volume_ml(mask, spacing):
    """
    Calculate mask volume in mL.
    spacing = voxel spacing in mm
    """

    voxel_volume_mm3 = (
        spacing[0] *
        spacing[1] *
        spacing[2]
    )

    voxel_count = np.sum(mask > 0)

    volume_mm3 = voxel_count * voxel_volume_mm3

    # 1 mL = 1000 mm3
    volume_ml = volume_mm3 / 1000.0

    return volume_ml


# ============================================================
# PROCESS PATIENTS
# ============================================================

print("\nStarting RV volume analysis...")

patients = sorted(
    {
        f.name.split("_")[0]
        for f in VAL_IMAGES.glob("*_0000.nii.gz")
    }
)

print("Patients found:", len(patients))

results = []

for patient in patients:

    row = {
        "Patient": patient
    }

    for phase in ["ED", "ES"]:

        img_file = VAL_IMAGES / f"{patient}_{phase}_0000.nii.gz"
        gt_file = VAL_GT / f"{patient}_{phase}.nii.gz"
        pred_file = PRED / f"{patient}_{phase}.nii.gz"

        if not img_file.exists():
            print("Missing image:", img_file)
            continue

        if not gt_file.exists():
            print("Missing GT:", gt_file)
            continue

        if not pred_file.exists():
            print("Missing prediction:", pred_file)
            continue

        # Load masks
        gt, spacing_gt = load_nii(gt_file)
        pred, spacing_pred = load_nii(pred_file)

        # Use GT spacing
        spacing = spacing_gt

        gt_volume = calculate_volume_ml(
            gt,
            spacing
        )

        pred_volume = calculate_volume_ml(
            pred,
            spacing
        )

        row[f"GT_{phase}V_ml"] = gt_volume
        row[f"Pred_{phase}V_ml"] = pred_volume

    # --------------------------------------------------------
    # EF
    # --------------------------------------------------------

    if (
        "GT_EDV_ml" in row and
        "GT_ESV_ml" in row and
        row["GT_EDV_ml"] > 0
    ):
        row["GT_EF_percent"] = (
            (row["GT_EDV_ml"] - row["GT_ESV_ml"])
            / row["GT_EDV_ml"]
        ) * 100

    if (
        "Pred_EDV_ml" in row and
        "Pred_ESV_ml" in row and
        row["Pred_EDV_ml"] > 0
    ):
        row["Pred_EF_percent"] = (
            (row["Pred_EDV_ml"] - row["Pred_ESV_ml"])
            / row["Pred_EDV_ml"]
        ) * 100

    results.append(row)


# ============================================================
# SAVE PATIENT LEVEL RESULTS
# ============================================================

df = pd.DataFrame(results)

csv_file = OUT / "RV_volume_results.csv"
df.to_csv(csv_file, index=False)

print("\nSaved:")
print(csv_file)

print("\nFirst rows:")
print(df.head())


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

variables = [
    ("EDV", "GT_EDV_ml", "Pred_EDV_ml"),
    ("ESV", "GT_ESV_ml", "Pred_ESV_ml"),
    ("EF", "GT_EF_percent", "Pred_EF_percent")
]

for name, gt_col, pred_col in variables:

    valid = df[[gt_col, pred_col]].dropna()

    gt = valid[gt_col].values
    pred = valid[pred_col].values

    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    bias = np.mean(pred - gt)
    correlation = np.corrcoef(gt, pred)[0, 1]

    summary_rows.append({
        "Measure": name,
        "N": len(valid),
        "GT_Mean": np.mean(gt),
        "Prediction_Mean": np.mean(pred),
        "MAE": mae,
        "RMSE": rmse,
        "Bias_Pred_minus_GT": bias,
        "Correlation": correlation
    })

summary = pd.DataFrame(summary_rows)

summary_file = OUT / "RV_volume_summary.csv"
summary.to_csv(summary_file, index=False)

print("\n==========================================")
print("RV VOLUME SUMMARY")
print("==========================================")

print(summary.to_string(index=False))

print("\nSaved:")
print(summary_file)


# ============================================================
# CHART 1 — EDV
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["GT_EDV_ml"],
    df["Pred_EDV_ml"],
    alpha=0.75
)

lims = [
    min(df["GT_EDV_ml"].min(), df["Pred_EDV_ml"].min()),
    max(df["GT_EDV_ml"].max(), df["Pred_EDV_ml"].max())
]

plt.plot(
    lims,
    lims,
    linestyle="--"
)

plt.xlabel("Ground Truth RVEDV (mL)")
plt.ylabel("Predicted RVEDV (mL)")
plt.title("RV End-Diastolic Volume: Ground Truth vs Prediction")
plt.tight_layout()

plt.savefig(
    OUT / "EDV_GT_vs_Prediction.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 2 — ESV
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["GT_ESV_ml"],
    df["Pred_ESV_ml"],
    alpha=0.75
)

lims = [
    min(df["GT_ESV_ml"].min(), df["Pred_ESV_ml"].min()),
    max(df["GT_ESV_ml"].max(), df["Pred_ESV_ml"].max())
]

plt.plot(
    lims,
    lims,
    linestyle="--"
)

plt.xlabel("Ground Truth RVESV (mL)")
plt.ylabel("Predicted RVESV (mL)")
plt.title("RV End-Systolic Volume: Ground Truth vs Prediction")
plt.tight_layout()

plt.savefig(
    OUT / "ESV_GT_vs_Prediction.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 3 — EF
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["GT_EF_percent"],
    df["Pred_EF_percent"],
    alpha=0.75
)

lims = [
    min(df["GT_EF_percent"].min(), df["Pred_EF_percent"].min()),
    max(df["GT_EF_percent"].max(), df["Pred_EF_percent"].max())
]

plt.plot(
    lims,
    lims,
    linestyle="--"
)

plt.xlabel("Ground Truth RVEF (%)")
plt.ylabel("Predicted RVEF (%)")
plt.title("RV Ejection Fraction: Ground Truth vs Prediction")
plt.tight_layout()

plt.savefig(
    OUT / "EF_GT_vs_Prediction.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


print("\n==========================================")
print("ALL VOLUME ANALYSIS COMPLETED")
print("==========================================")
print(OUT)