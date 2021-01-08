import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import chain
from pathlib import Path

from jointclassifier.data import CombinedMNIST, MNIST
from jointclassifier.models import Encoder, Embedder, Joiner


class Trainer:
    def __init__(self, x_latent_dim, y_latent_dim, hidden_dim,
                 in_channels, multipler, img_dim, num_classes,
                 lr, epochs, batch_size, grad_clip, weight_decay,
                 checkpoint_dir, data_dir):
        super().__init__()
        self.encoder = Encoder(in_channels, multipler, x_latent_dim, hidden_dim, img_dim)
        self.classifier = Encoder(in_channels, multipler, num_classes, hidden_dim, img_dim, classifier=True)
        self.embedder = Embedder(num_classes, y_latent_dim, hidden_dim)
        self.joiner = Joiner(x_latent_dim, y_latent_dim, hidden_dim)
        self.models = {'encoder': self.encoder, 'classifier': self.classifier, 'embedder': self.embedder, 'joiner': self.joiner}

        self.num_classes = num_classes
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        Path('/'.join(checkpoint_dir.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer_disc = torch.optim.Adam(chain(*[m.parameters() for name, m in self.models.items() if name != 'classifier']), lr=lr, weight_decay=weight_decay)
        self.optimizer_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_loader = DataLoader(CombinedMNIST(data_dir + '/train.csv', num_classes), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(MNIST(data_dir + '/train.csv'), batch_size=batch_size, shuffle=False)

        for model in self.models.values():
            model.to(self.device)

    def fit(self):
        pbar = tqdm(range(self.epochs))
        iterations = 0
        for epoch in pbar:
            disc_losses, cls_losses = 0, 0
            for model in self.models.values():
                model.train()
            for i, (x1, x2, y) in enumerate(self.train_loader):
                iterations += 1

                #########################
                # Training Discriminator#
                #########################
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                # normal x_y
                y_oh = torch.zeros(y.size(0), self.num_classes, device=y.device)
                y_oh.scatter_(1, y.long(), 1)
                x_y = self.joiner(self.encoder(x1), self.embedder(y_oh))

                # classified x_y
                y_hat = self.classifier(x2)
                x_y_hat = self.joiner(self.encoder(x2), self.embedder(y_hat.detach()))

                # update params, clip heuristic enforce 1lipschitz
                disc_loss = x_y_hat.mean() - x_y.mean()
                self.optimizer_disc.zero_grad()
                disc_loss.backward(retain_graph=True)
                if self.grad_clip is not None:
                    for _, model in self.models.items():
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip, norm_type='inf')
                self.optimizer_disc.step()
                disc_losses += disc_loss.item()

                ######################
                # Training Classifier#
                ######################
                x_y_hat = self.joiner(self.encoder(x2), self.embedder(y_hat))
                cls_loss = -x_y_hat.mean()
                self.optimizer_cls.zero_grad()
                cls_loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clip, norm_type='inf')
                self.optimizer_cls.step()
                cls_losses += cls_loss.item()

                pbar.set_postfix({
                    'Iteration': iterations,
                    'Discriminator Loss': disc_loss.item(),
                    'Classifier Loss': cls_loss.item(),
                    'x_y': x_y.mean().item(),
                    'x_y_hat': x_y_hat.mean().item()})

            # Testing classifier
            for model in self.models.values():
                model.eval()
            corrects, samples = 0, 0
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    y_hat = self.classifier(x)
                corrects += (y_hat.argmax(1) == y).sum()
                samples += y.size(0)
            tqdm.write(
                f'Epoch {epoch + 1}/{self.epochs}, \
                    Discriminator Loss: {disc_losses / (i + 1):.3f}, \
                        Classifier Loss: {cls_losses / (i + 1):.3f}, \
                        Accuracy: {corrects * 100 / samples:.3f}%'
            )
            torch.save({name: model.state_dict() for name, model in self.models.items()}, self.checkpoint_dir)
