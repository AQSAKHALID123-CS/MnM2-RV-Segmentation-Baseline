from pathlib import Path
import nibabel as nib
import numpy as np

VAL_DIR = Path(r"F:\MnM2_Validation\imagesTs")

bad_files = [
    "139_ED_0000.nii.gz",
    "139_ES_0000.nii.gz",
    "206_ED_0000.nii.gz",
    "206_ES_0000.nii.gz",
    "256_ED_0000.nii.gz",
    "256_ES_0000.nii.gz",
    "270_ED_0000.nii.gz",
    "270_ES_0000.nii.gz",
    "279_ED_0000.nii.gz",
    "279_ES_0000.nii.gz",
    "320_ED_0000.nii.gz",
    "320_ES_0000.nii.gz"
]

for name in bad_files:
    path = VAL_DIR / name
    print("Fixing:", name)

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    affine = img.affine.copy()

    # Original voxel spacing
    direction_part = affine[:3, :3]
    spacing = np.linalg.norm(direction_part, axis=0)

    # Normalize direction matrix
    direction = direction_part / spacing

    # Orthogonalize direction matrix
    U, _, Vt = np.linalg.svd(direction)
    corrected_direction = U @ Vt

    # Make it right-handed
    if np.linalg.det(corrected_direction) < 0:
        U[:, -1] *= -1
        corrected_direction = U @ Vt

    # Rebuild affine while preserving spacing and origin
    corrected_affine = np.eye(4)
    corrected_affine[:3, :3] = corrected_direction @ np.diag(spacing)
    corrected_affine[:3, 3] = affine[:3, 3]

    repaired = nib.Nifti1Image(
        data,
        corrected_affine,
        img.header
    )

    nib.save(repaired, str(path))

    print("  Fixed.")

print("\nDONE - 12 validation files repaired.")