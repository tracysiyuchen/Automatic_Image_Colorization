import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ImageDataset
from src.util import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, batch_size=128, lr=0.001, epochs=10, device="cpu"):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        for inputs, targets in train_loader:
            inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
            targets = targets.permute(0, 3, 1, 2).float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # outputs_lab = outputs
            # real_image = targets
            l_channel = inputs[:, :1, :, :]  # Selecting the first channel
            outputs_lab = torch.cat([l_channel, outputs], dim=1)
            real_image = torch.cat([l_channel, targets], dim=1)

            loss = self.criterion(outputs_lab, real_image)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if epoch % 50 == 0:
                for i in range(outputs_lab.size(0)):
                    output_image = outputs_lab[i].detach().numpy()
                    target_image = real_image[i].detach().numpy()
                    psnr_val = psnr(output_image, target_image, data_range=2)
                    ssim_val = ssim(output_image, target_image, data_range=2, multichannel=True, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val

        average_loss = total_loss / len(train_loader)
        if epoch % 50 == 0:
            average_psnr = total_psnr / len(train_loader.dataset)
            average_ssim = total_ssim / len(train_loader.dataset)
            print(f'Training Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')
        else:
            print(f'Training Loss: {average_loss:.4f}')


    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
                targets = targets.permute(0, 3, 1, 2).float().to(self.device)
                outputs = self.model(inputs)
                # outputs_lab = outputs
                # real_image = targets
                l_channel = inputs[:, :1, :, :]  # Selecting the first channel
                outputs_lab = torch.cat([l_channel, outputs], dim=1)
                real_image = torch.cat([l_channel, targets], dim=1)
                loss = self.criterion(outputs_lab, real_image)
                total_loss += loss.item()
                if epoch % 50 == 0:
                    for i in range(outputs_lab.size(0)):
                        output_image = outputs_lab[i].cpu().numpy()
                        target_image = real_image[i].cpu().numpy()
                        psnr_val = psnr(output_image, target_image, data_range=2)
                        ssim_val = ssim(output_image, target_image, data_range=2, multichannel=True, channel_axis=0)
                        total_psnr += psnr_val
                        total_ssim += ssim_val

        average_loss = total_loss / len(val_loader)
        if epoch % 50 == 0:
            average_psnr = total_psnr / len(val_loader.dataset)
            average_ssim = total_ssim / len(val_loader.dataset)
            print(f'Validation Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')
        else:
            print(f'Validation Loss: {average_loss:.4f}')

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
                targets = targets.permute(0, 3, 1, 2).float().to(self.device)
                outputs = self.model(inputs)
                # outputs_lab = outputs
                # real_image = targets
                l_channel = inputs[:, :1, :, :]
                outputs_lab = torch.cat([l_channel, outputs], dim=1)
                real_image = torch.cat([l_channel, targets], dim=1)

                loss = self.criterion(outputs_lab, real_image)
                total_loss += loss.item()
                # predicted = lab_to_rgb(inputs, outputs)[0]
                # plt.imshow(predicted)
                # plt.axis('off')
                # plt.show()
                for i in range(outputs_lab.size(0)):
                    output_image = outputs_lab[i].cpu().numpy()
                    target_image = real_image[i].cpu().numpy()
                    psnr_val = psnr(output_image, target_image, data_range=2)
                    ssim_val = ssim(output_image, target_image, data_range=2, multichannel=True, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val
        average_loss = total_loss / len(test_loader)
        average_psnr = total_psnr / len(test_loader.dataset)
        average_ssim = total_ssim / len(test_loader.dataset)
        print(f'Test Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')

    def train(self, train_data, val_data, test_data=None):
        train_dataset = ImageDataset(*train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = ImageDataset(*val_data)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        test_loader = None
        if test_data is not None:
            test_dataset = ImageDataset(*test_data)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print("Start training..")
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}:')
            self.train_one_epoch(train_loader, epoch)
            self.validate_one_epoch(val_loader, epoch)

        if test_loader is not None:
            self.test(test_loader)
