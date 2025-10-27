import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class DRAEMTrainDataset(Dataset):

    def __init__(self, root_path, img_dirs, anomaly_source_path='data/dtd/images', resize_shape=(1024, 1024)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_path = root_path
        self.resize_shape=resize_shape

        self.img_paths = []
        for i, img_dir in enumerate(img_dirs):
            img_names = os.listdir(os.path.join(root_path, img_dir))
            self.img_paths += [os.path.join(root_path, img_dir, img_name) for img_name in img_names if ".jpg" in img_name]

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.img_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 2
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.bool_)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
            # image = self.rot(image=image)

        image = np.array(image).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1)) * 255

        augmented_image = augmented_image * 255
        return augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.img_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {"label": anomaly_mask,
                  'image': augmented_image, 'valid': has_anomaly}

        return sample