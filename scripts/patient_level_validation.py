from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path(r"F:\MnM2_Validation\validation_metrics.csv")

df = pd.read_csv(INPUT)

# -----------------------------
# ED / ES statistics
# -----------------------------

print("\n==============================")
print("PHASE-WISE RESULTS")
print("==============================")

for phase in ["ED", "ES"]:

    d = df[df["Phase"] == phase]

    print(f"\n{phase}:")
    print("Cases :", len(d))

    print("Dice:")
    print("  Mean   :", d["Dice"].mean())
    print("  Median :", d["Dice"].median())
    print("  Std    :", d["Dice"].std())
    print("  Min    :", d["Dice"].min())
    print("  Max    :", d["Dice"].max())

    print("HD95:")
    print("  Mean   :", d["HD95_mm"].mean())
    print("  Median :", d["HD95_mm"].median())
    print("  Std    :", d["HD95_mm"].std())
    print("  Min    :", d["HD95_mm"].min())
    print("  Max    :", d["HD95_mm"].max())


# -----------------------------
# Patient-level aggregation
# -----------------------------

patient_df = (
    df.groupby("Patient")
      .agg(
          Mean_Dice=("Dice", "mean"),
          Median_Dice=("Dice", "median"),
          Mean_HD95=("HD95_mm", "mean"),
          Median_HD95=("HD95_mm", "median")
      )
      .reset_index()
)

# Save patient-level results
output = Path(r"F:\MnM2_Validation\patient_level_results.csv")
patient_df.to_csv(output, index=False)


print("\n==============================")
print("PATIENT-LEVEL RESULTS")
print("==============================")

print("Patients:", len(patient_df))

print("\nMean patient Dice:")
print(patient_df["Mean_Dice"].mean())

print("Median patient Dice:")
print(patient_df["Mean_Dice"].median())

print("Std patient Dice:")
print(patient_df["Mean_Dice"].std())

print("\nMean patient HD95:")
print(patient_df["Mean_HD95"].mean())

print("Median patient HD95:")
print(patient_df["Mean_HD95"].median())

print("Std patient HD95:")
print(patient_df["Mean_HD95"].std())


# -----------------------------
# Worst 5 patients
# -----------------------------

worst5 = patient_df.sort_values(
    "Mean_Dice",
    ascending=True
).head(5)

print("\n==============================")
print("5 WORST PATIENTS")
print("==============================")

print(worst5.to_string(index=False))

worst5.to_csv(
    r"F:\MnM2_Validation\worst5_patients.csv",
    index=False
)

print("\nSaved:")
print(output)
print(r"F:\MnM2_Validation\worst5_patients.csv")