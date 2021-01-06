import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, multipler, latent_dim, hidden_dim, img_dim, classifier=False):
        super().__init__()
        self.classifier = classifier
        self.conv1 = nn.Conv2d(in_channels, multipler, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(multipler, multipler * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(multipler * 2, multipler * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(multipler * 4, multipler * 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(multipler * 4, multipler * 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(multipler)
        self.bn2 = nn.BatchNorm2d(multipler * 2)
        self.bn3 = nn.BatchNorm2d(multipler * 4)
        self.bn4 = nn.BatchNorm2d(multipler * 4)
        self.fc1 = nn.Linear((img_dim // (2 ** 2)) ** 2 * multipler * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.classifier:
            x = F.softmax(x, -1)
        return x


class Embedder(nn.Module):
    def __init__(self, num_classes, latent_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, y):
        return self.embed(y)


class Joiner(nn.Module):
    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_latent_dim + y_latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x_y = torch.cat([x, y], 1)
        x_y = F.relu(self.fc1(x_y))
        return self.fc2(x_y)
