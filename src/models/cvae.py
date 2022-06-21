import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.distributions import Normal


class Decoder(torch.nn.Module):
    def __init__(self, n_classes, cond_embedding_dim, latent_dim):
        super().__init__()
        self.label_conditioned_generator = torch.nn.Sequential(
            torch.nn.Embedding(n_classes, cond_embedding_dim),
            torch.nn.Linear(cond_embedding_dim, 16)
        )

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 4*4*512),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
            torch.nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
            torch.nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
            torch.nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64*1, 1, 4, 2, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        mu, log_std, label = inputs
        dist = Normal(mu, torch.exp(log_std + 1e-3))
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(dist.rsample())
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        return image


class Encoder(torch.nn.Module):
    def __init__(self, n_classes, cond_embedding_dim, embedding_dim):
        super().__init__()
        self.label_condition_disc = torch.nn.Sequential(
            torch.nn.Embedding(n_classes, cond_embedding_dim),
            torch.nn.Linear(cond_embedding_dim, 1 * 128 * 128)
        )

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
            torch.nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
            torch.nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),
            torch.nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(4608, 2 * embedding_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(embedding_dim, embedding_dim),
        )
        self.log_std = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 1, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)
        return self.mu(output), self.log_std(output)


class CVAEModel(pl.LightningModule):
    def __init__(self, n_classes, embedding_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(n_classes, embedding_dim, latent_dim)
        self.dec = Decoder(n_classes, embedding_dim, latent_dim)
        self.latent_dim = latent_dim
        self.loss = torch.nn.MSELoss()
        self.lr = 1e-4

    def forward(self, label: int):
        noise_mu = torch.randn(1, self.latent_dim).to(self.device)
        noise_log_std = torch.randn(1, self.latent_dim).to(self.device)
        res = self.gen((noise_mu, noise_log_std, torch.IntTensor([[label]]).to(self.device)))
        return res

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)

    def training_step(self, train_batch, batch_idx):
        real_labels = train_batch["lab"].long()
        real_data = train_batch["img"]

        mu, log_std = self.enc((real_data, real_labels))
        restored_img = self.dec((mu, log_std, real_labels))

        loss = self.loss(real_data, restored_img)
        self.log("vae_loss", loss)

        return loss

    def sample_random(self, label, size=1):
        noise_mu = torch.randn(size, self.latent_dim).to(self.device)
        noise_log_std = torch.randn(size, self.latent_dim).to(self.device)
        res = self.gen((noise_mu, noise_log_std, torch.IntTensor([[label]] * size).to(self.device)))
        return res

    def training_epoch_end(self, training_step_outputs):
        noise_mu = torch.randn(1, self.latent_dim).to(self.device)
        noise_log_std = torch.randn(1, self.latent_dim).to(self.device)

        res = self.dec((noise_mu, noise_log_std, torch.IntTensor([[1]]).to(self.device)))
        plt.imshow(res[0].cpu().detach().numpy().squeeze(0), cmap="Greys_r")
        plt.show()