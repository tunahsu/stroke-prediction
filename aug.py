import numpy as np
import os
import random
import PIL
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from volumentations import Compose, RandomGamma, Rotate



def read_data_file(filepath):
    slices = []
    for scan in sorted(os.listdir(filepath)):
        slice = np.asarray(PIL.Image.open(os.path.join(filepath, scan)))
        slices.append(slice)
    slices = np.array(slices)
    return slices


def get_augmentation(patch_size, value):
    return Compose([
#         Rotate((0, 0), (0, 0), (15, 15), p=1),
#         Flip(0, p=1),
#         Flip(1, p=1),
#         Flip(2, p=0.5),
#         RandomRotate90((1, 2), p=1),
        RandomGamma(gamma_limit=(value, value + 1), p=1),
    ], p=1.0)


def process_scan(path):
    volume = read_data_file(path).astype(np.uint8)

    values = [90, 95, 100, 105]
    for i in range(len(values)):
        new_path = path.replace('ROI_Dataset', 'aug_roi_dataset') + '_{}'.format(i + 1)
        os.makedirs(new_path)

        aug = get_augmentation((volume.shape[0], volume.shape[1], volume.shape[2]), values[i])

        new_volume = np.random.randint(0, 255, size=(volume.shape[0], volume.shape[1], volume.shape[2]), dtype=np.uint8)

        # without mask
        data = {'image': volume}
        aug_data = aug(**data)
        new_volume = aug_data['image']
        

        for j in range(new_volume.shape[0]):
            img = PIL.Image.fromarray(new_volume[j].astype(np.uint8))
            img_path = new_path + '\\{0:03d}.jpg'.format(j + 1)
            # img.save(img_path)
            cv2.imwrite(img_path, new_volume[j])

    return volume



normal_scan_paths = [
    os.path.join(os.getcwd(), "ROI_Dataset/LOW", x)
    for x in os.listdir("ROI_Dataset/LOW")
]
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "ROI_Dataset/HIGH", x)
    for x in os.listdir("ROI_Dataset/HIGH")
]

normal_scans = np.array([process_scan(path) for path in tqdm(normal_scan_paths)])
abnormal_scans = np.array([process_scan(path) for path in tqdm(abnormal_scan_paths)])


