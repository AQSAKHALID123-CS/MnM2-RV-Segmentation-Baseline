from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_DIR = Path(r"F:\MnM2\MnM2\dataset")
SPLIT_DIR = Path(r"F:\MnM2_Split")
GT_DIR = Path(r"F:\MnM2_Validation\labelsVal")

GT_DIR.mkdir(parents=True, exist_ok=True)

val_patients = [
    x.strip()
    for x in (SPLIT_DIR / "val.txt").read_text().splitlines()
    if x.strip()
]

count = 0

for patient in val_patients:
    patient_dir = DATASET_DIR / patient

    for phase in ["ED", "ES"]:

        gt_file = patient_dir / f"{patient}_SA_{phase}_gt.nii"

        if not gt_file.exists():
            print("Missing:", gt_file)
            continue

        case_id = f"{patient}_{phase}"
        output_file = GT_DIR / f"{case_id}.nii.gz"

        gt = nib.load(str(gt_file))
        gt_data = gt.get_fdata()

        # Original M&Ms-2 label:
        # 0 = background
        # 1 = LV
        # 2 = Myocardium
        # 3 = RV
        rv_mask = (gt_data == 3).astype(np.uint8)

        mask_img = nib.Nifti1Image(
            rv_mask,
            gt.affine,
            gt.header
        )

        mask_img.set_data_dtype(np.uint8)

        nib.save(mask_img, str(output_file))

        count += 1

print("\nDone!")
print("Validation patients:", len(val_patients))
print("GT masks created:", count)
print("Expected:", len(val_patients) * 2)