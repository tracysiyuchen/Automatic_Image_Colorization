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
        total_images = 0  # Keep track of total images processed

        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            total_images += batch_size  # Update total images processed

            inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
            targets = targets.permute(0, 3, 1, 2).float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            l_channel = inputs[:, :1, :, :]  # Selecting the first channel
            outputs_lab = torch.cat([l_channel, outputs], dim=1)
            real_image = torch.cat([l_channel, targets], dim=1)

            loss = self.criterion(outputs_lab, real_image)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                for i in range(outputs_lab.size(0)):
                    output_image = outputs_lab[i].detach().cpu().numpy()
                    target_image = real_image[i].detach().cpu().numpy()
                    psnr_val = psnr(output_image, target_image, data_range=2)
                    ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val

        average_loss = total_loss / len(train_loader)
        if epoch % 50 == 0:
            average_psnr = total_psnr / total_images
            average_ssim = total_ssim / total_images
            print(f'Training Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')
        else:
            print(f'Training Loss: {average_loss:.4f}')



    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_images = 0  # Keep track of total images processed

        with torch.no_grad():
            for inputs, targets in val_loader:
                batch_size = inputs.size(0)
                total_images += batch_size  # Update total images processed

                inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
                targets = targets.permute(0, 3, 1, 2).float().to(self.device)
                outputs = self.model(inputs)

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
                        ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                        total_psnr += psnr_val
                        total_ssim += ssim_val

        average_loss = total_loss / len(val_loader)
        if epoch % 50 == 0:
            average_psnr = total_psnr / total_images
            average_ssim = total_ssim / total_images
            print(f'Validation Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')
        else:
            print(f'Validation Loss: {average_loss:.4f}')


    def test(self, test_loader,model_name):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_images = 0
        grayscale_images_list = []
        predicted_images_list = []
        desired_output_list = []

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
                total_images += outputs_lab.size(0)

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
                    ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val

                if len(grayscale_images_list) < 5:
                    predicted_images = lab_to_rgb(l_channel, outputs)
                    desired_outputs = lab_to_rgb(l_channel, targets)

                    grayscale_images_list.extend(l_channel.squeeze().cpu().numpy())
                    predicted_images_list.extend(predicted_images)
                    desired_output_list.extend(desired_outputs)

            plt.figure(figsize=(10, 10))
            for i in range(5):
                # Display the grayscale image on the first column
                plt.subplot(5, 3, 3*i + 1)
                plt.imshow(grayscale_images_list[i], cmap='gray')
                plt.axis('off')
                if i == 0:
                  plt.title('Grayscale')

                # Display the predicted image on the second column
                plt.subplot(5, 3, 3*i + 2)
                plt.imshow(predicted_images_list[i])
                plt.axis('off')
                if i == 0:
                  plt.title(f'{model_name} Result')

                # Display the desired output on the third column
                plt.subplot(5, 3, 3*i + 3)
                plt.imshow(desired_output_list[i])
                plt.axis('off')
                if i == 0:
                  plt.title('Desired Result')

        plt.tight_layout()
        plt.savefig(f'output_figure_{model_name}.png', dpi=500, format='png')

        average_loss = total_loss / len(test_loader)
        average_psnr = total_psnr / total_images
        average_ssim = total_ssim / total_images
        print(f'Test Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')

    def train(self, train_data, val_data, test_data=None):
        model_name = self.model.__class__.__name__

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
            self.test(test_loader,model_name)