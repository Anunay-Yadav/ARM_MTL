import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextNet2D(nn.Module):
    def __init__(self, n_in, n_out, n_hid, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size, padding)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.conv3 = nn.Conv2d(n_hid, n_out, kernel_size, padding)
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_in, n_out, n_hid, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size, padding)
        self.conv2 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.conv3 = nn.Conv2d(n_hid, n_hid, kernel_size, padding)
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
        self.bn3 = nn.BatchNorm2d(n_hid)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x