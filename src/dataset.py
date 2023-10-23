from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, gray_images, ab_images, transform=None):
        self.gray_images = gray_images
        self.ab_images = ab_images
        self.transform = transform

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_image = self.gray_images[idx]/128 - 1
        ab_image = self.ab_images[idx]/128 - 1
        return gray_image, ab_image
