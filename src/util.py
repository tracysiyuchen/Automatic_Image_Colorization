import numpy as np
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab, lab2rgb
import torch


def split_data(gray_images, ab_images, splits=[0.7, 0.1, 0.2]):
    train_size, val_size, test_size = splits
    indices = np.arange(len(gray_images))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (train_size + val_size)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size_adjusted, random_state=42)
    train_data = (gray_images[train_indices], ab_images[train_indices])
    val_data = (gray_images[val_indices], ab_images[val_indices])
    test_data = (gray_images[test_indices], ab_images[test_indices])
    return train_data, val_data, test_data


def load_data(gray_path, ab_path, num_images=None):
    images_gray = np.load(gray_path)
    images_lab = np.load(ab_path)
    min_len = min(len(images_gray), len(images_lab))
    if num_images is not None:
        if num_images > min_len:
            num_images = min_len
    else:
        num_images = min_len
    images_gray = images_gray[:num_images]
    images_lab = images_lab[:num_images]
    return images_gray, images_lab


def lab_to_rgb(L, ab):
    L = (L + 1.) * 50
    ab = ab * 128
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
