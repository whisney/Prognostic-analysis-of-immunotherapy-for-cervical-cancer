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

An Excel file (metadata/data_all.xlsx), which stores clinical information for model training, is required and is presented in the format below:

| ID | Landm | age | tnm | Past | Initial | Number | Type | ALB | LM | ANC | PLT | SCC | Label | Progress | PFS | death | OS |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ID001 | 1 | 68 | 2 | 7 | 0 | 4 | 1 | 40.7 | 0.51 | 3.43 | 262 | 3.6 | 0 | 1 | 6 | 1 | 11 |
| ID002 | 0 | 76 | 3 | 0 | 4 | 2 | 1 | 44.2 | 0.33 | 3.3 | 203 | 25.1 | 1 | 0 | 29 | 0 | 29 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Before starting training, ROI region needs to be cropped and data needs to be divided into training sets and verification sets. You can run this step with the following command:
```
python crop_rois.py
python Stratified_split.py
```

## Model training
For training therapeutic response prediction model, you can run:
```
python train_Response.py
```
For training survival analysis model, you can run:
```
python train_Survival.py --task OS
```
For training progress-free survival analysis model, you can run:
```
python train_Survival.py --task PFS
```
