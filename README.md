\# M\&Ms-2 Right Ventricular Segmentation Baseline



This repository contains the implementation, evaluation scripts, commands, and results for a baseline right ventricular (RV) segmentation study on the M\&Ms-2 cardiac MRI dataset.



\## Objective



The objective is to establish a reproducible baseline for automatic right ventricular segmentation from short-axis cardiac cine MRI at end-diastole (ED) and end-systole (ES).



The baseline uses nnU-Net v2 with a 3D full-resolution configuration.



\## Dataset



The study uses the M\&Ms-2 (Multi-Centre, Multi-Vendor \& Multi-Disease Cardiac Image Segmentation) dataset.



The dataset contains multi-centre cardiac MRI data with different vendors, scanners, field strengths, and cardiac phases.



For this baseline:



\- Short-axis (SA) MRI was used.

\- ED and ES phases were used.

\- The original multiclass annotations were converted to an RV-only binary segmentation task.

\- Original RV label: 3.

\- Background: 0.

\- Converted RV mask: 1.



The dataset itself is not included in this repository because of data-access and redistribution restrictions.



\## Patient-Level Data Split



A strict patient-level split was used:



| Partition | Patients |

|---|---:|

| Training | 251 |

| Validation | 55 |

| Test | 54 |

| Total | 360 |



Split ratio: 70% / 15% / 15%



Random seed: 42



Both ED and ES phases of the same patient were kept within the same partition to prevent patient-level data leakage.



\## Preprocessing



nnU-Net v2 automatic planning and preprocessing were used.



The 3D full-resolution configuration was selected because the input MRI volumes contain multiple short-axis slices and inter-slice spatial context can contribute to RV segmentation.



The median resampled spacing was approximately:



\- Slice direction: 9.6 mm

\- In-plane: 1.25 × 1.25 mm



The automatically determined patch size was:



`12 × 256 × 320`



\## Model



Baseline architecture:



\- nnU-Net v2

\- 3D full-resolution

\- PlainConvUNet configuration

\- Fold 0

\- Single MRI input channel



The baseline was trained initially using Fold 0 to establish the complete preprocessing, training, inference, and evaluation pipeline.



The reported validation performance was obtained on the independent 55-patient validation set.



\## Training



Training commands are provided in:



`commands/training.txt`



Environment information is provided in:



\- `requirements.txt`

\- `environment.yml`



\## Evaluation



The model was evaluated on 110 validation cases corresponding to 55 patients:



\- 55 ED cases

\- 55 ES cases



Metrics:



\- Dice Similarity Coefficient (Dice)

\- 95th percentile Hausdorff Distance (HD95)



Both case-level and patient-level results are provided.



\## Overall Validation Results



| Metric | Result |

|---|---:|

| Mean Dice | 0.9223 |

| Median Dice | 0.9299 |

| Mean HD95 | 4.5765 mm |

| Median HD95 | 2.1063 mm |

| Minimum Dice | 0.6415 |

| Maximum Dice | 0.9769 |



\## ED and ES Performance



\### ED



\- Mean Dice: 0.9350

\- Mean HD95: 4.7953 mm



\### ES



\- Mean Dice: 0.9096

\- Mean HD95: 4.3577 mm



The baseline achieved higher Dice at ED than ES.



\## Failure Analysis



The five lowest-performing patients based on mean patient-level Dice were:



| Patient | Mean Dice | Mean HD95 |

|---|---:|---:|

| 261 | 0.803964 | 10.966417 mm |

| 349 | 0.828058 | 10.237147 mm |

| 186 | 0.857864 | 5.908231 mm |

| 84 | 0.861252 | 10.034832 mm |

| 233 | 0.875908 | 9.600000 mm |



Patient 261 at ES was the poorest individual case:



\- Dice: 0.6415

\- HD95: 20.75 mm



Qualitative overlays are provided in:



`results/Qualitative/`



\## Volume Analysis



RV end-diastolic volume (EDV), end-systolic volume (ESV), and ejection fraction (EF) were estimated from both reference and predicted masks.



Results and plots are provided in:



`results/Volume\_Analysis/`



\## Metadata-Based Subgroup Analysis



Where metadata were available, performance was examined across:



\- Disease

\- Vendor

\- Scanner

\- Magnetic-field strength



The original metadata did not contain a VIEW field; therefore, view-wise analysis was not performed.



Results are provided in:



`results/Subgroup\_Analysis/`



\## Repository Structure



```text

MnM2\_GitHub/

│

├── README.md

├── requirements.txt

├── environment.yml

│

├── commands/

│   ├── training.txt

│   └── inference.txt

│

├── scripts/

│   ├── make\_all\_ppt\_visuals.py

│   ├── make\_overlays.py

│   ├── make\_ppt\_images.py

│   ├── make\_subgroup\_analysis.py

│   └── make\_volume\_analysis.py

│

└── results/

&#x20;   ├── validation\_metrics.csv

&#x20;   ├── Patient\_Level/

&#x20;   ├── Volume\_Analysis/

&#x20;   ├── Subgroup\_Analysis/

&#x20;   ├── Figures/

&#x20;   └── Qualitative/

