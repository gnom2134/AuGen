import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class Generator(torch.nn.Module):
    def __init__(self, n_classes, embedding_dim, latent_dim):
        super().__init__()
        self.label_conditioned_generator = torch.nn.Sequential(
            torch.nn.Embedding(n_classes, embedding_dim),
            torch.nn.Linear(embedding_dim, 16)
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
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        return image


class Discriminator(torch.nn.Module):
    def __init__(self, n_classes, embedding_dim):
        super(Discriminator, self).__init__()

        self.label_condition_disc = torch.nn.Sequential(
            torch.nn.Embedding(n_classes, embedding_dim),
            torch.nn.Linear(embedding_dim, 1 * 128 * 128)
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
            torch.nn.Linear(4608, 1)
        )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 1, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)
        return output


class CGANModel(pl.LightningModule):
    def __init__(self, n_classes, embedding_dim, latent_dim):
        super().__init__()
        self.gen = Generator(n_classes, embedding_dim, latent_dim)
        self.disc = Discriminator(n_classes, embedding_dim)
        self.latent_dim = latent_dim
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

    def forward(self):
        noise_vector = torch.randn(1, self.latent_dim).to(self.device)
        res = self.gen((noise_vector, torch.IntTensor([[1]]).to(self.device)))
        return res

    def configure_optimizers(self):
        return torch.optim.Adam(self.disc.parameters(), lr=1e-3), torch.optim.Adam(self.gen.parameters(), lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        d_opt, g_opt = self.optimizers()

        real_labels = train_batch["lab"].long()
        real_data = train_batch["img"]
        noise_vector = torch.randn(real_data.size(0), self.latent_dim).to(self.device)

        fake_target = torch.zeros((real_data.size(0), 1)).to(self.device)
        real_target = torch.ones((real_data.size(0), 1)).to(self.device)

        D_real_loss = self.loss(self.disc((real_data, real_labels)), real_target)

        generated_image = self.gen((noise_vector, real_labels))
        output = self.disc((generated_image.detach(), real_labels))
        D_fake_loss = self.loss(output, fake_target)

        D_total_loss = (D_real_loss + D_fake_loss) / 2
        d_opt.zero_grad()
        self.manual_backward(D_total_loss)
        d_opt.step()

        d_opt.zero_grad()
        g_opt.zero_grad()
        G_loss = self.loss(self.disc((generated_image, real_labels)), real_target)
        self.manual_backward(G_loss)
        g_opt.step()

        self.log("discriminator_loss", D_total_loss)
        self.log("generator_loss", G_loss)

        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step(G_loss)

        return D_total_loss

    def sample_random(self, label, size=1):
        noise_vector = torch.randn(size, self.latent_dim).to(self.device)
        res = self.gen((noise_vector, torch.IntTensor([[label]] * size).to(self.device)))
        return res

    def training_epoch_end(self, training_step_outputs):
        noise_vector = torch.randn(1, self.latent_dim).to(self.device)
        res = self.gen((noise_vector, torch.IntTensor([[1]]).to(self.device)))
        plt.imshow(res[0].cpu().detach().numpy().squeeze(0), cmap="Greys_r")
        plt.show()
