\# Results



This folder documents the evaluation outputs of the M\&Ms-2 right ventricular (RV) segmentation baseline.



\## Validation Dataset



\- Validation patients: 55

\- Validation cases: 110

\- Each patient contributes ED and ES phases.

\- Patient-level separation was maintained between training and validation data.

\- The validation set was not used during model training.



\## Overall Segmentation Results



| Metric | Value |

|---|---:|

| Mean Dice | 0.9223 |

| Median Dice | 0.9299 |

| Mean HD95 (mm) | 4.5765 |

| Median HD95 (mm) | 2.1063 |

| Minimum Dice | 0.6415 |

| Maximum Dice | 0.9769 |

| Maximum HD95 (mm) | 20.7504 |



\## ED vs ES



\### End-Diastolic (ED)



\- Mean Dice: 0.9350

\- Median Dice: 0.9481

\- Mean HD95: 4.7953 mm



\### End-Systolic (ES)



\- Mean Dice: 0.9096

\- Median Dice: 0.9240

\- Mean HD95: 4.3577 mm



The model achieved higher mean Dice at ED than ES. HD95 was slightly lower at ES, indicating that Dice and boundary-distance metrics do not necessarily change in the same direction.



\## Worst Five Patients



| Patient | Mean Dice | Mean HD95 (mm) |

|---|---:|---:|

| 261 | 0.803964 | 10.966417 |

| 349 | 0.828058 | 10.237147 |

| 186 | 0.857864 | 5.908231 |

| 84 | 0.861252 | 10.034832 |

| 233 | 0.875908 | 9.600000 |



The poorest individual case was patient 261 at ES, with Dice = 0.6415 and HD95 = 20.75 mm.



\## Patient-Level Reporting



Patient-level results are provided in:



\- `patient\_level\_results.csv`

\- `worst5\_patients.csv`



\## Volume Analysis



RV EDV, ESV, and EF were estimated from both reference and predicted RV masks.



The volume-analysis outputs are stored separately in:



`Volume\_Analysis/`



Files include:



\- `RV\_volume\_results.csv`

\- `RV\_volume\_summary.csv`

\- `EDV\_GT\_vs\_Prediction.png`

\- `ESV\_GT\_vs\_Prediction.png`

\- `EF\_GT\_vs\_Prediction.png`



\## Subgroup Analysis



Metadata-based analyses were performed for disease, vendor, scanner, and magnetic-field strength where metadata were available.



Outputs are stored in:



`Subgroup\_Analysis/`



No view-wise analysis was performed because the available metadata did not contain a VIEW field.



\## Qualitative Analysis



Qualitative overlays of reference and predicted RV masks were generated for the five poorest-performing patients.



The corresponding visualizations are stored in:



`PPT\_Images/Worst5\_Overlays/`



\## Reproducibility



The complete training and inference commands are documented in:



\- `commands/training.txt`

\- `commands/inference.txt`



The Python environment is documented in:



\- `requirements.txt`

\- `environment.yml`

