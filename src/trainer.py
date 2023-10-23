import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageDataset
from util import *
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, batch_size=128, lr=0.001, epochs=10, device="cpu"):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.unsqueeze(1).float().to(self.device)
            targets = targets.permute(0, 3, 1, 2).float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Training Loss: {average_loss:.4f}')

    def validate_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.unsqueeze(1).float().to(self.device)
                targets = targets.permute(0, 3, 1, 2).float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        average_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {average_loss:.4f}')

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.unsqueeze(1).float().to(self.device)
                targets = targets.permute(0, 3, 1, 2).float().to(self.device)
                outputs = self.model(inputs)
                outputs_lab = torch.cat([inputs, outputs], dim=1)
                real_image = torch.cat([inputs, targets], dim=1)
                loss = self.criterion(outputs_lab, real_image)
                total_loss += loss.item()
                predicted = lab_to_rgb(inputs, outputs)[0]
                plt.imshow(predicted)
                plt.axis('off')
                plt.show()
        average_loss = total_loss / len(test_loader)
        print(f'Test Loss: {average_loss:.4f}')

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
            self.train_one_epoch(train_loader)
            self.validate_one_epoch(val_loader)

        if test_loader is not None:
            self.test(test_loader)

