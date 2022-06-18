import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(VAE, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))

        self.latent_dim = latent_dim
        if n_classes == 2:
            self.classes_dim = 1
        else:
            self.classes_dim = n_classes

        self.encoder_fc1 = nn.Linear(32 * 32 * 32, self.latent_dim)
        self.encoder_fc2 = nn.Linear(32 * 32 * 32, self.latent_dim)
        self.Sigmoid = nn.Sigmoid()

        self.decoder_fc = nn.Linear(self.latent_dim + self.classes_dim, 32 * 32 * 32)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self, mean, log_var):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(log_var)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def encoder(self, x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        log_std = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, log_std)
        return z, mean, log_std

    def decoder(self, z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 32, 32)
        out3 = self.decoder_deconv(out3)
        return out3


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class CVAEGANModel(pl.LightningModule):

    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.vae = VAE(latent_dim, n_classes)
        self.discriminator = Discriminator(1)
        
        self.latent_dim = latent_dim
        if n_classes == 2:
            self.classes_dim = 1
        else:
            self.classes_dim = n_classes

        self.classifier = Discriminator(self.classes_dim)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, z, y):
        z = torch.hstack((z, y))
        return self.vae.decoder(z)

    def loss_function(self, recon_x, x, mean, log_std):
        MSE = nn.MSELoss()(recon_x, x)
        var = torch.pow(torch.exp(log_std), 2)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        return MSE + KLD

    def classifier_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.classifier(x)
        c_loss = self.criterion(output.unsqueeze(1), y)

        self.log('classifier loss', c_loss)

        return c_loss

    def discriminator_step(self, x):
        x = x.to(self.device)

        output = self.discriminator(x)
        real_label = torch.ones(x.shape[0], device=self.device)
        fake_label = torch.zeros(x.shape[0], device=self.device)
        loss_real = self.criterion(output, real_label)

        z = torch.randn(x.shape[0], self.vae.latent_dim + self.classes_dim, device=self.device)
        fake_data = self.vae.decoder(z)
        output = self.discriminator(fake_data.detach())
        loss_fake = self.criterion(output, fake_label)

        loss = (loss_real + loss_fake) / 2
        self.log('discriminator loss', loss)

        return loss

    def vae_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        z, mean, log_std = self.vae.encoder(x)

        z = torch.cat([z, y], 1)
        recon_data = self.vae.decoder(z)
        vae_loss1 = self.loss_function(recon_data, x, mean, log_std)

        output = self.discriminator(recon_data)
        real_label = torch.ones(x.shape[0], device=self.device)
        vae_loss2 = self.criterion(output, real_label)

        output = self.classifier(recon_data)
        real_label = y
        vae_loss3 = self.criterion(output.unsqueeze(1), real_label)

        v_loss = vae_loss1 + vae_loss2 + vae_loss3
        self.log('vae loss', v_loss)

        return v_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y = batch['img'], batch['lab']

        if self.classes_dim == 1:
            label_dummy = y
        else:
            label_dummy = torch.zeros((X.shape[0], self.classes_dim)).to(self.device)
            label_dummy[torch.arange(X.shape[0]), y] = 1

        if optimizer_idx == 0:
            loss = self.classifier_step(X, label_dummy)
        if optimizer_idx == 1:
            loss = self.discriminator_step(X)
        if optimizer_idx == 2:
            loss = self.vae_step(X, label_dummy)

        self.log('sum loss', loss)

        return loss

    def configure_optimizers(self):
        c_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        v_optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0002)
        return c_optimizer, d_optimizer, v_optimizer

    def training_epoch_end(self, training_step_outputs):
        z = torch.randn(1, self.vae.latent_dim + 1, device=self.device)
        res = self.vae.decoder(z)

        cur = res[0].cpu().detach().numpy().squeeze(0)
        plt.imshow(cur, cmap="Greys_r")
        plt.show()
        # torch.save(self.state_dict(),
        #            "./drive/MyDrive/dl-project/CVAEGAN_on_siim_dataset.state")
