import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ImageDataset
from src.models import MobileNet, Critic
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.util import *
import matplotlib.pyplot as plt

class GAN_Trainer:
    def __init__(self, learning_rate=0.0002, lambda_recon=100, lambda_gp=10, lambda_r1=10,
                 batch_size=128, lr=0.001, epochs=10, device="cpu"):
        self.generator = MobileNet()
        self.critic = Critic()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.L1Loss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.generator = self.generator.to(device)
        self.critic = self.critic.to(device)

    def train_one_epoch(self, train_loader, epoch, device):
        self.generator.train()
        self.critic.train()
        total_gen_loss = 0
        total_critic_loss = 0
        total_psnr = 0
        total_ssim = 0

        for conditioned_images, real_images in train_loader:
            conditioned_images = conditioned_images.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
            real_images = real_images.permute(0, 3, 1, 2).float().to(self.device)

            self.optimizer_C.zero_grad()

            fake_images = self.generator(conditioned_images)
            fake_logits = self.critic(fake_images, conditioned_images)
            real_logits = self.critic(real_images, conditioned_images)

            loss_C = real_logits.mean() - fake_logits.mean()

            alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
            interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
            interpolated_logits = self.critic(interpolated, conditioned_images)
            grad_outputs = torch.ones_like(interpolated_logits, device=device)
            gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs,
                                            create_graph=True, retain_graph=True)[0]
            gradients = gradients.view(len(gradients), -1)
            gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            r1_reg = gradients.pow(2).sum(1).mean()

            loss_C += self.lambda_gp * gradients_penalty + self.lambda_r1 * r1_reg
            loss_C.backward()
            self.optimizer_C.step()
            total_critic_loss += loss_C.item()

            self.optimizer_G.zero_grad()
            fake_images = self.generator(conditioned_images)
            recon_loss = self.recon_criterion(fake_images, real_images)
            recon_loss.backward()
            self.optimizer_G.step()
            total_gen_loss += recon_loss.item()

            l_channel = conditioned_images[:, :1, :, :]
            outputs_lab = torch.cat([l_channel, fake_images], dim=1)
            real_image = torch.cat([l_channel, real_images], dim=1)

            if epoch % 50 == 0:
                fake_images = fake_images.detach()
                for i in range(fake_images.size(0)):
                    output_image = outputs_lab[i].detach().cpu().numpy()
                    target_image = real_image[i].detach().cpu().numpy()
                    psnr_val = psnr(output_image, target_image, data_range=2)
                    ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val

        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_critic_loss = total_critic_loss / len(train_loader)

        if epoch % 50 == 0:
            avg_psnr = total_psnr / len(train_loader.dataset)
            avg_ssim = total_ssim / len(train_loader.dataset)
            print(
                f"Epoch {epoch}: Generator Loss: {avg_gen_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        else:
            print(f"Epoch {epoch}: Generator Loss: {avg_gen_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    def validate_one_epoch(self, val_loader, epoch):
        self.generator.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0

        with torch.no_grad():
            for conditioned_images, real_images in val_loader:
                conditioned_images = conditioned_images.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
                real_images = real_images.permute(0, 3, 1, 2).float().to(self.device)

                fake_images = self.generator(conditioned_images)
                loss = self.recon_criterion(fake_images, real_images)
                total_loss += loss.item()

                l_channel = conditioned_images[:, :1, :, :]
                outputs_lab = torch.cat([l_channel, fake_images], dim=1)
                real_image = torch.cat([l_channel, real_images], dim=1)

                if epoch % 50 == 0:
                    for i in range(fake_images.size(0)):
                        output_image = outputs_lab[i].detach().cpu().numpy()
                        target_image = real_image[i].detach().cpu().numpy()
                        psnr_val = psnr(output_image, target_image, data_range=2)
                        ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                        total_psnr += psnr_val
                        total_ssim += ssim_val

        average_loss = total_loss / len(val_loader)
        if epoch % 50 == 0:
            average_psnr = total_psnr / len(val_loader.dataset)
            average_ssim = total_ssim / len(val_loader.dataset)
            print(f'Validation Loss: {average_loss:.4f}, PSNR: {average_psnr:.4f}, SSIM: {average_ssim:.4f}')
        else:
            print(f'Validation Loss: {average_loss:.4f}')

    def test(self, test_loader, model_name):
        self.generator.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_images = 0
        grayscale_images_list = []
        predicted_images_list = []
        desired_output_list = []

        with torch.no_grad():
            for conditioned_images, real_images in test_loader:
                conditioned_images = conditioned_images.unsqueeze(1).repeat(1, 3, 1, 1).float().to(self.device)
                real_images = real_images.permute(0, 3, 1, 2).float().to(self.device)

                fake_images = self.generator(conditioned_images)
                loss = self.recon_criterion(fake_images, real_images)
                total_loss += loss.item()

                l_channel = conditioned_images[:, :1, :, :]
                outputs_lab = torch.cat([l_channel, fake_images], dim=1)
                real_image = torch.cat([l_channel, real_images], dim=1)
                total_images += outputs_lab.size(0)


                for i in range(fake_images.size(0)):
                    output_image = outputs_lab[i].detach().cpu().numpy()
                    target_image = real_image[i].detach().cpu().numpy()
                    psnr_val = psnr(output_image, target_image, data_range=2)
                    ssim_val = ssim(output_image, target_image, data_range=2, channel_axis=0)
                    total_psnr += psnr_val
                    total_ssim += ssim_val

                predicted_images = lab_to_rgb(l_channel, fake_images)
                desired_outputs = lab_to_rgb(l_channel, real_images)

                grayscale_images_list.extend(l_channel.squeeze().cpu().numpy())
                predicted_images_list.extend(predicted_images)
                desired_output_list.extend(desired_outputs)
                if len(grayscale_images_list) >= 5:
                    break

            plt.figure(figsize=(10, 10))
            for i in range(5):
                # Display the grayscale image on the first column
                plt.subplot(5, 3, 3 * i + 1)
                plt.imshow(grayscale_images_list[i], cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Grayscale')

                # Display the predicted image on the second column
                plt.subplot(5, 3, 3 * i + 2)
                plt.imshow(predicted_images_list[i])
                plt.axis('off')
                if i == 0:
                    plt.title(f'{model_name} Result')

                # Display the desired output on the third column
                plt.subplot(5, 3, 3 * i + 3)
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
        model_name = "GAN"

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
            self.train_one_epoch(train_loader, epoch, self.device)
            self.validate_one_epoch(val_loader, epoch)

        if test_loader is not None:
            self.test(test_loader,model_name)