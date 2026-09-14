from pathlib import Path
import SimpleITK as sitk
import numpy as np

VAL_DIR = Path(r"F:\MnM2_Validation\imagesTs")

bad = []

for f in sorted(VAL_DIR.glob("*.nii.gz")):
    try:
        img = sitk.ReadImage(str(f))
        direction = np.array(img.GetDirection()).reshape(3, 3)

        # Check orthonormality
        error = np.max(np.abs(direction.T @ direction - np.eye(3)))

        if error > 1e-5:
            bad.append((f.name, error))
            print("BAD:", f.name, "error =", error)

    except Exception as e:
        bad.append((f.name, str(e)))
        print("BAD:", f.name, "ERROR:", e)

print("\nTotal validation images:", len(list(VAL_DIR.glob("*.nii.gz"))))
print("Bad files:", len(bad))