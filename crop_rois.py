from Nii_utils import NiiDataRead, NiiDataWrite
import os
import glob
import numpy as np
from skimage.measure import *
from utils import remove_small_areas

data_dir = 'Nii_Data'
save_dir = 'Nii_Data_ROIs'

for ID in os.listdir(data_dir):
    print(ID)
    mask_path_list = glob.glob(os.path.join(data_dir, ID, '*mask.nii.gz'))
    for mask_path in mask_path_list:
        basename = os.path.basename(mask_path).rstrip('_mask.nii.gz')
        print(basename)
        os.makedirs(os.path.join(save_dir, ID, basename), exist_ok=True)
        img, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, ID, '{}.nii.gz'.format(basename)))
        mask, _, _, _ = NiiDataRead(mask_path, as_type=np.uint8)
        mask = remove_small_areas(mask, min_area=10)
        if mask.max() == 1:
            regions_area = []
            connect_regions = label(mask, connectivity=2, background=0)
            props = regionprops(connect_regions)
            for n in range(len(props)):
                regions_area.append(props[n].area)
            index = np.argsort(np.array(regions_area))
            index = np.flip(index)
            for i in range(len(props)):
                index_one = index[i]
                filled_value = props[index_one].label
                mask_one = (connect_regions == filled_value).astype(np.uint8)
                z, x, y = mask_one.nonzero()
                z1 = z.min()
                z2 = z.max() + 1
                x1 = x.min()
                x2 = x.max() + 1
                y1 = y.min()
                y2 = y.max() + 1
                img_roi = img[z1: z2, x1: x2, y1: y2]
                mask_roi = mask_one[z1: z2, x1: x2, y1: y2]
                NiiDataWrite(os.path.join(save_dir, ID, basename, '{}_{}.nii.gz'.format(basename, i)), img_roi, spacing, origin, direction)
                NiiDataWrite(os.path.join(save_dir, ID, basename, '{}_mask_{}.nii.gz'.format(basename, i)), mask_roi, spacing, origin, direction, as_type=np.uint8)
