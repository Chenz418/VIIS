import random

import torch
import os
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from ZC_image2image.Data_Augment import brightness_aug, poisson_noise, add_gaussian_noise, add_salt_and_pepper_noise
from skimage import transform
import cv2


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

def get_transform(method=Image.BICUBIC):
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

class I2IDatasetBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.size = size

        self.image_paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if
                   os.path.basename(f).endswith('D.png')]
        self._length = len(self.image_paths)

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.Norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_name = os.path.basename(self.image_paths[i])

        # --- Load and augment RGB input image ---
        image_rgb = Image.open(image_path)
        contrast_factor = random.uniform(0.1, 1.0)
        image_rgb = ImageEnhance.Contrast(image_rgb).enhance(contrast_factor)
        image_rgb = np.array(image_rgb).astype(np.uint8)

        crop_size = min(image_rgb.shape[:2])
        x1 = np.random.randint(0, image_rgb.shape[1] - crop_size + 1)
        image_rgb = image_rgb[:, x1:x1 + crop_size, :]

        # Apply custom augmentations
        image_rgb = brightness_aug(image_rgb, 3 + random.random() * 7)
        image_rgb = poisson_noise(image_rgb)
        image_rgb = add_gaussian_noise(image_rgb)

        # Resize and normalize to [-1, 1]
        image_rgb = cv2.resize(image_rgb, (self.size, self.size))
        image_rgb = (image_rgb / 127.5 - 1.0).astype(np.float32)

        # --- Load infrared image ---
        infrared_path = image_path.replace("rgbs", "infrareds")
        image_ir = Image.open(infrared_path).convert("L").convert("RGB")
        image_ir = np.array(image_ir).astype(np.uint8)
        image_ir = image_ir[:, x1:x1 + crop_size, :]
        image_ir = cv2.resize(image_ir, (self.size, self.size))
        image_ir = (image_ir / 127.5 - 1.0).astype(np.float32)

        # --- Load supervised ground truth (RGB) image ---
        image_gt = Image.open(image_path)
        image_gt = np.array(image_gt)[:, x1:x1 + crop_size, :].astype(np.uint8)
        image_gt = cv2.resize(image_gt, (self.size, self.size))
        image_gt = (image_gt / 127.5 - 1.0).astype(np.float32)

        return image_gt, image_rgb, image_ir, image_name

class I2IDatasetVal(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.image_paths = [data_root + '/' + os.path.basename(f) for f in os.listdir(data_root)]
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.Norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self._length

    def _load_and_preprocess(self, path):
        """Loads, center-crops, resizes, and normalizes an image."""
        image = Image.open(path).convert("RGB")
        image = np.array(image).astype(np.uint8)

        h, w = image.shape[:2]
        crop_squ = min(h, w)
        image = image[(h - crop_squ) // 2:(h + crop_squ) // 2,
                      (w - crop_squ) // 2:(w + crop_squ) // 2]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_name = os.path.basename(self.image_paths[i])

        rgb_image = self._load_and_preprocess(image_path)
        infrared_path = image_path.replace('rgbs', 'infrareds')
        infrared_image= self._load_and_preprocess(infrared_path)
        
        return rgb_image, rgb_image, infrared_image, image_name

class I2IDatasetTrain(I2IDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class I2IDatasetValidation(I2IDatasetVal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
