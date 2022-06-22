import torch
import pytorch_lightning as pl


class ResNetModel(pl.LightningModule):
    def __init__(self, resnet_model):
        super().__init__()
        self.resnet_model = resnet_model
        self.resnet_model.fc = torch.nn.Linear(self.resnet_model.fc.in_features, 1)
        self.resnet_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = 1e-4

    def forward(self, x):
        return torch.sigmoid(self.resnet_model(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        y = train_batch["lab"]
        x = train_batch["img"]
        y_pred = self.resnet_model(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        y = val_batch["lab"]
        x = val_batch["img"]
        y_pred = torch.sigmoid(self.resnet_model(x))
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss)
        return loss
