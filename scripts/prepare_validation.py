from pathlib import Path
import nibabel as nib

DATASET_DIR = Path(r"F:\MnM2\MnM2\dataset")
SPLIT_DIR = Path(r"F:\MnM2_Split")
VAL_DIR = Path(r"F:\MnM2_Validation")
imagesTs = VAL_DIR / "imagesTs"

imagesTs.mkdir(parents=True, exist_ok=True)

val_patients = [
    x.strip()
    for x in (SPLIT_DIR / "val.txt").read_text().splitlines()
    if x.strip()
]

count = 0

for patient in val_patients:
    patient_dir = DATASET_DIR / patient

    for phase in ["ED", "ES"]:
        image_file = patient_dir / f"{patient}_SA_{phase}.nii"

        if not image_file.exists():
            print("Missing:", image_file)
            continue

        case_id = f"{patient}_{phase}"
        output_file = imagesTs / f"{case_id}_0000.nii.gz"

        img = nib.load(str(image_file))
        nib.save(img, str(output_file))

        count += 1

print("\nDone!")
print("Validation patients:", len(val_patients))
print("Validation cases:", count)
print("Expected:", len(val_patients) * 2)