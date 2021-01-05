import torch
from torch.
from tqdm import tqdm
from models import Encoder, Embedder, Joiner


class Trainer:
    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim,
                 in_channels, multipler, img_dim, num_classes,
                 lr, epochs):
        super().__init__()
        self.num_classes = num_classes
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.encoder = Encoder(in_channels, multipler, x_latent_dim, hidden_dim, img_dim)
        self.classifier = Encoder(in_channels, multipler, num_classes, hidden_dim, img_dim)
        self.embedder = Embedder(num_classes, y_latent_dim, hidden_dim)
        self.joiner = Joiner(x_latent_dim, y_latent_dim, hidden_dim)

    def training_step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = y.scatter_(1, self.num_classes, 1)

        # original x_y
        x_y = self.joiner(x, self.embedder(y))

        # generated x_y
        y_hat = self.classifier(x)
        x_y_hat = self.joiner(x, self.embedder(y_hat))

        loss = x_y.mean() - x_y_hat.mean()
        return loss

    def fit(self, train_loader, test_loader):
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            for batch in train_loader:
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()
