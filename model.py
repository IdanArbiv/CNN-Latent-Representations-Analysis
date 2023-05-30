import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Recon_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
#         self.pool = nn.MaxPool2d(2, 2, return_indices=True)
#         self.unpool = nn.MaxUnpool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         z_1, indices_1 = self.pool(F.relu(self.conv1(x)))
#         z_2, indices_2 = self.pool(F.relu(self.conv2(z_1)))
#         x = self.deconv2(F.relu(self.unpool(z_2, indices_2)))
#         x = self.deconv1(F.relu(self.unpool(x, indices_1)))
#         y = torch.flatten(z_2, 1) # flatten all dimensions except batch
#         y = F.relu(self.fc1(y))
#         y = F.relu(self.fc2(y))
#         y = self.fc3(y)
#         return y, x, z_1, z_2


class Recon_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x, z = None, i = None):
        z_1, indices_1 = self.pool(F.relu(self.conv1(x)))
        if z == "1":
            z_1 = self.take_specific_feature(i, z_1)
        z_2, indices_2 = self.pool(F.relu(self.conv2(z_1)))
        if z == "2":
            z_2 = self.take_specific_feature(i, z_2)

        x = self.deconv2(F.relu(self.unpool(z_2, indices_2)))
        x = self.deconv1(F.relu(self.unpool(x, indices_1)))
        y = torch.flatten(z_2, 1)  # flatten all dimensions except batch
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y, x

    def take_specific_feature(self, i, z_1):
        # Set all channels except the specified one to 0
        z_1[:, :i - 1, :, :] = 0
        z_1[:, i:, :, :] = 0
        return z_1

