from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

METADATA = Path(r"F:\MnM2\MnM2\dataset_information.csv")
METRICS = Path(r"F:\MnM2_Validation\validation_metrics.csv")
OUT = Path(r"F:\MnM2_Validation\Subgroup_Analysis")

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading metadata...")

metadata = pd.read_csv(
    METADATA,
    low_memory=False
)

metrics = pd.read_csv(METRICS)

print("Metadata shape:", metadata.shape)
print("Validation cases:", len(metrics))

# ============================================================
# PREPARE PATIENT IDs
# ============================================================

# Validation Patient column
metrics["Patient_ID"] = pd.to_numeric(
    metrics["Patient"],
    errors="coerce"
).astype("Int64")

# Metadata subject code
metadata["Patient_ID"] = pd.to_numeric(
    metadata["SUBJECT_CODE"],
    errors="coerce"
).astype("Int64")

# ============================================================
# KEEP ONE METADATA ROW PER PATIENT
# ============================================================

patient_metadata = (
    metadata[
        [
            "Patient_ID",
            "DISEASE",
            "VENDOR",
            "SCANNER",
            "FIELD"
        ]
    ]
    .dropna(subset=["Patient_ID"])
    .drop_duplicates(subset=["Patient_ID"])
)

print("Unique metadata patients:", len(patient_metadata))

# ============================================================
# MERGE
# ============================================================

df = metrics.merge(
    patient_metadata,
    on="Patient_ID",
    how="left"
)

print("\nMatched metadata:")
print(
    df[
        ["Patient_ID", "DISEASE", "VENDOR", "SCANNER", "FIELD"]
    ].head(10).to_string(index=False)
)

# ============================================================
# CHECK MATCHING
# ============================================================

missing = df["DISEASE"].isna().sum()

print("\nMissing metadata matches:", missing)

if missing > 0:
    print("\nPatients without metadata:")
    print(
        df.loc[
            df["DISEASE"].isna(),
            "Patient_ID"
        ].unique()
    )

# ============================================================
# FUNCTION FOR GROUP ANALYSIS
# ============================================================

def subgroup_analysis(data, column, name):

    print("\n======================================")
    print(name)
    print("======================================")

    result = (
        data
        .groupby(column, dropna=False)
        .agg(
            Patients=("Patient_ID", "nunique"),
            Cases=("Dice", "count"),
            Mean_Dice=("Dice", "mean"),
            Median_Dice=("Dice", "median"),
            Mean_HD95_mm=("HD95_mm", "mean"),
            Median_HD95_mm=("HD95_mm", "median")
        )
        .reset_index()
        .sort_values("Mean_Dice", ascending=False)
    )

    print(result.to_string(index=False))

    result.to_csv(
        OUT / f"{name}_analysis.csv",
        index=False
    )

    return result


# ============================================================
# DISEASE ANALYSIS
# ============================================================

disease = subgroup_analysis(
    df,
    "DISEASE",
    "Disease"
)

# ============================================================
# VENDOR ANALYSIS
# ============================================================

vendor = subgroup_analysis(
    df,
    "VENDOR",
    "Vendor"
)

# ============================================================
# SCANNER ANALYSIS
# ============================================================

scanner = subgroup_analysis(
    df,
    "SCANNER",
    "Scanner"
)

# ============================================================
# FIELD ANALYSIS
# ============================================================

field = subgroup_analysis(
    df,
    "FIELD",
    "Field"
)

# ============================================================
# COMBINED PATIENT-LEVEL METADATA TABLE
# ============================================================

df.to_csv(
    OUT / "Validation_With_Metadata.csv",
    index=False
)

# ============================================================
# CHART FUNCTION
# ============================================================

def make_dice_chart(result, category, filename, title):

    # Only show groups with at least 1 patient
    result = result.copy()

    plt.figure(figsize=(9, 6))

    plt.bar(
        result[category].astype(str),
        result["Mean_Dice"]
    )

    plt.ylabel("Mean Dice")
    plt.xlabel(category)
    plt.title(title)

    plt.ylim(0, 1)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    for i, value in enumerate(result["Mean_Dice"]):

        plt.text(
            i,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CREATE CHARTS
# ============================================================

make_dice_chart(
    disease,
    "DISEASE",
    "Disease_Dice.png",
    "RV Segmentation Performance by Disease Group"
)

make_dice_chart(
    vendor,
    "VENDOR",
    "Vendor_Dice.png",
    "RV Segmentation Performance by MRI Vendor"
)

make_dice_chart(
    field,
    "FIELD",
    "Field_Dice.png",
    "RV Segmentation Performance by Magnetic Field Strength"
)

# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("SUBGROUP ANALYSIS COMPLETED")
print("======================================")

print("Output folder:")
print(OUT)

print("\nGenerated files:")

for file in sorted(OUT.iterdir()):
    print(file.name)