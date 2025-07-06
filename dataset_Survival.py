import torch
import pickle
from Nii_utils import NiiDataRead
import pandas as pd
from volumentations import *
import os
from torch.utils.data import Dataset
from skimage import transform
import random
import nnet_survival_pytorch

mean_std_dict = {'age': [54.04210526315789, 9.308644301173064], 'tnm': [2.7473684210526317, 0.9230204542710688],
                       'Number': [2.4789473684210526, 0.8316622210137334], 'ALB': [41.42315789473684, 3.8742252911579884],
                       'LM': [1.2303447368421052, 0.6642932450108431], 'ANC': [4.828684210526316, 3.0029660522229227],
                       'PLT': [253.22368421052633, 92.64365781744056], 'SCC': [14.356921052631579, 20.09026634556826]}

def Extract_clinical_features(metadata_df, ID):
    Clinical_features = []

    Clinical_features.append(torch.from_numpy(
        np.array([int(metadata_df.loc[metadata_df.ID == ID, 'Landm'].values[0])])))

    Past = int(metadata_df.loc[metadata_df.ID == ID, 'Past'].values[0])
    if Past == 0:
        Clinical_features.append(torch.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0, 0])))
    elif Past == 1:
        Clinical_features.append(torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 0, 0])))
    elif Past == 2:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 1, 0, 0, 0, 0, 0])))
    elif Past == 3:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 1, 0, 0, 0, 0])))
    elif Past == 4:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 1, 0, 0, 0])))
    elif Past == 5:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 0, 0])))
    elif Past == 6:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 1, 0])))
    elif Past == 7:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 0, 1])))
    else:
        print('Past error!')

    Initial = int(metadata_df.loc[metadata_df.ID == ID, 'Initial'].values[0])
    if Initial == 0:
        Clinical_features.append(torch.from_numpy(np.array([1, 0, 0, 0, 0, 0])))
    elif Initial == 1:
        Clinical_features.append(torch.from_numpy(np.array([0, 1, 0, 0, 0, 0])))
    elif Initial == 2:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 1, 0, 0, 0])))
    elif Initial == 3:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 1, 0, 0])))
    elif Initial == 4:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 1, 0])))
    elif Initial == 5:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 0, 0, 0, 1])))
    else:
        print('Initial error!')

    Type = int(metadata_df.loc[metadata_df.ID == ID, 'Type'].values[0])
    if Type == 0:
        Clinical_features.append(torch.from_numpy(np.array([1, 0, 0])))
    elif Type == 1:
        Clinical_features.append(torch.from_numpy(np.array([0, 1, 0])))
    elif Type == 2:
        Clinical_features.append(torch.from_numpy(np.array([0, 0, 1])))
    else:
        print('Type error!')

    for feature_name in ['age', 'tnm', 'Number', 'ALB', 'LM', 'ANC', 'PLT', 'SCC']:
        feature_value = float(metadata_df.loc[metadata_df.ID == ID, feature_name].values[0])
        feature_value = (feature_value - mean_std_dict[feature_name][0]) / mean_std_dict[feature_name][1]
        Clinical_features.append(torch.from_numpy(np.array([feature_value])))
    return torch.cat(Clinical_features)

class Dataset_ST(Dataset):
    def __init__(self, data_dir, split_path, metadata_path, breaks, data_set='train', task='OS', augment=True):
        self.data_dir = data_dir
        self.task = task
        self.breaks = breaks
        split_data = pickle.load(open(split_path, 'rb'))
        self.ID_list = split_data[data_set]

        self.augment = augment

        self.metadata_df = pd.read_excel(metadata_path)
        df_name = list(self.metadata_df['name_pinyin'])

        self.transforms = Compose([
            RotatePseudo2D(axes=(1, 2), limit=(-30, 30), interpolation=3, value=0, mask_value=0, p=0.3),
            ElasticTransformPseudo2D(alpha=20, sigma=10, alpha_affine=2, value=0, p=0.3)
        ])

        self.len = len(self.ID_list)

    def __getitem__(self, idx):
        ID = self.ID_list[idx]
        if self.task == 'OS':
            event_time = int(self.metadata_df.loc[self.metadata_df.ID == ID, 'OS'].values[0])
            event_indicator = int(self.metadata_df.loc[self.metadata_df.ID == ID, 'death'].values[0])
        elif self.task == 'PFS':
            event_time = int(self.metadata_df.loc[self.metadata_df.ID == ID, 'PFS'].values[0])
            event_indicator = int(self.metadata_df.loc[self.metadata_df.ID == ID, 'Progress'].values[0])

        label = nnet_survival_pytorch.make_surv_array(np.array([event_time]), np.array([event_indicator]), self.breaks)

        Clinical_features = Extract_clinical_features(self.metadata_df, ID)
        if self.augment:
            Clinical_features += torch.randn(*Clinical_features.size()) * 0.05

        modality_sign = []
        imgs = []
        for modality in ['CT_P', 'CT_Z', 'MR_P', 'MR_Z']:
            if os.path.exists(os.path.join(self.data_dir, ID, modality)):
                modality_sign.append(1)
            else:
                modality_sign.append(0)
            for i in range(4):
                if os.path.exists(os.path.join(self.data_dir, ID, modality, '{}_{}.nii.gz'.format(modality, i))):
                    img, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, modality, '{}_{}.nii.gz'.format(modality, i)))
                    mask, _, _, _ = NiiDataRead(os.path.join(self.data_dir, ID, modality, '{}_mask_{}.nii.gz'.format(modality, i)))
                    if 'CT' in modality:
                        img = np.clip(img, -120, 180)
                        img = (img + 120) / 300
                    else:
                        mean = np.mean(img[mask > 0])
                        std = np.std(img[mask > 0])
                        img = (img - mean) / std
                    if self.augment:
                        auged = self.transforms(image=img, mask=mask)
                        img = auged['image']
                        mask = auged['mask']
                    img = transform.resize(img, (16, 96, 96), order=0, mode='constant', clip=False,
                                           preserve_range=True, anti_aliasing=False)
                    mask = transform.resize(mask, (16, 96, 96), order=0, mode='constant', clip=False,
                                            preserve_range=True, anti_aliasing=False)
                else:
                    img = np.zeros((16, 96, 96))
                    mask = np.zeros((16, 96, 96))
                imgs.append(np.concatenate([img[np.newaxis, ...], mask[np.newaxis, ...]], axis=0))
        if self.augment:
            if sum(modality_sign) > 1:
                index_for_1 = [i for i, val in enumerate(modality_sign) if val == 1]
                random_index = random.choice(index_for_1)
                index_for_potential_aug = list(set(index_for_1).difference(set([random_index])))
                for index_ in index_for_potential_aug:
                    if random.randint(0, 2) == 0:
                        modality_sign[index_] = 0

        imgs = torch.from_numpy(np.concatenate(imgs, axis=0))
        modality_sign = torch.from_numpy(np.array(modality_sign))
        event_time = torch.tensor(event_time).unsqueeze(0)
        event_indicator = torch.tensor(event_indicator).unsqueeze(0)
        label = torch.from_numpy(label)[0]
        return imgs, modality_sign, Clinical_features, event_time, event_indicator, label

    def __len__(self):
        return self.len