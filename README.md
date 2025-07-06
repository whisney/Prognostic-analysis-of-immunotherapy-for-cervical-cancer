# Prognostic-analysis-of-immunotherapy-for-cervical-cancer
Development and Validation of a Deep Learning Model for Predicting Immunotherapy Response and Survival Outcomes in Patients with Advanced Cervical Cancer

## Requirements
* python 3.6
* pytorch 1.5+
* pandas
* scikit-learn
* scikit-image
* tensorboardX
* SimpleITK
* lifelines

## Data preparation
First, the ROIs of the tumors in non-enhanced CT (CT_P), contrast-enhanced CT (CT_Z), non-enhanced MR (MR_P), and contrast-enhanced MR (MR_Z) images are segmented manually or automatically. The original images and segmentation masks should be placed in the **Nii_Data** folder in the following form and name:
```
- Nii_Data
  - ID001
    - CT_P.nii.gz
    - CT_P_mask.nii.gz
    - CT_Z.nii.gz
    - CT_Z_mask.nii.gz
    - MR_P.nii.gz
    - MR_P_mask.nii.gz
    - MR_Z.nii.gz
    - MR_Z_mask.nii.gz
  - ID002
    - ...
  - ...
- README.md
- LICENSE
- ...
```

An Excel file (metadata/Response.xlsx), which stores clinical information for therapeutic response prediction, is required and is presented in the format below:

| ID | Landm | age | tnm | Past | Initial | Number | Type | ALB | LM | ANC | PLT | SCC | Label |
| :-----: | :----: | :----: | :----: | :-----:  :-----:  :-----:  :-----:  :-----:  :-----:  :-----:  :-----:  :-----:  :-----: 
| ID001 | 53 | male | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| ID002 | 51 | female | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
