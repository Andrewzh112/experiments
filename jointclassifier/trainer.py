import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import chain
from pathlib import Path

from jointclassifier.data import CombinedMNIST
from jointclassifier.models import Encoder, Embedder, Joiner


class Trainer:
    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim,
                 in_channels, multipler, img_dim, num_classes,
                 lr, epochs, batch_size, grad_clip,
                 checkpoint_dir, data_dir):
        super().__init__()
        self.encoder = Encoder(in_channels, multipler, x_latent_dim, hidden_dim, img_dim)
        self.classifier = Encoder(in_channels, multipler, 1, hidden_dim, img_dim)
        self.embedder = Embedder(y_latent_dim, hidden_dim)
        self.joiner = Joiner(x_latent_dim, y_latent_dim, hidden_dim)
        self.models = {'encoder': self.encoder, 'classifier': self.classifier, 'embedder': self.embedder, 'joiner': self.joiner}

        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(chain(*[m.parameters() for m in self.models.values()]), lr=lr)
        self.train_loader = DataLoader(CombinedMNIST(data_dir + '/train.csv'), batch_size=batch_size,
                                       shuffle=True)

    def training_step(self, batch):
        x1, x2, y = batch
        x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

        # normal x_y
        x_y = self.joiner(self.encoder(x1), self.embedder(y))

        # classified x_y
        y_hat = self.classifier(x2)
        x_y_hat = self.joiner(self.encoder(x2), self.embedder(y_hat))

        loss = x_y_hat.mean() - x_y.mean()
        return loss

    def fit(self):
        pbar = tqdm(range(self.epochs))
        iterations = 0
        for epoch in pbar:
            losses = 0
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                for _, model in self.models.items():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type='inf')
                self.optimizer.step()
                iterations += i
                pbar.set_postfix({'Loss': round(loss.item(), 3), 'Iteration': iterations})
                losses += loss.item()
            tqdm.write(
                f'Epoch {epoch + 1}/{self.epochs}, \
                    Train Loss: {losses / (i + 1):.3f}'
            )
            torch.save({model.state_dict() for name, model in self.models.items()}, self.checkpoint_dir)
