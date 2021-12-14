import pytorch_lightning as pl
import torch

from typing import Any
from torch import Tensor
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import MSELoss
from torchvision import transforms


class RacingF1Detector(pl.LightningModule):

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)
        self.loss_function = MSELoss()
        self.losses = []

        self.conv_layers = Sequential(
            Conv2d(3, 64, self.hparams.kernel_size, 4, 1, device=self.device),
            ReLU(),
            Conv2d(64, 128, self.hparams.kernel_size, padding=1, device=self.device),
            ReLU(),
            Conv2d(128, 256, self.hparams.kernel_size, padding=1, device=self.device),
            Flatten()
        )

        self.regression_layers = Sequential(
            Linear(256 * 160 * 90, 128, device=self.device),
            ReLU(),
            Linear(128, 64, device=self.device),
            ReLU(),
            Linear(64, 32, device=self.device),
            ReLU(),
            Linear(32, 4, device=self.device),
            Sigmoid()
        )


    def forward(self, x: Tensor, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        out = self.conv_layers(x)
        out = self.regression_layers(out)
        return out


    def training_step(self, batch: dict, batch_idx: int) -> Tensor:

        inputs = transforms.ToTensor()(batch['img'])
        labels = torch.Tensor([label.tolist() for label in batch['bounding_box']])
        
        logits = self.forward(inputs)
        labels = torch.IntTensor(torch.einsum('ij->ji', labels))
        loss = self.loss_function(logits, labels)
        self.losses.append(loss)
        
        avg_loss = torch.stack([i for i in self.losses]).mean()
        # self.logger.experiment.add_scalar("Loss", avg_loss, self.current_epoch) # GOOGLE COLAB
        
        
        return loss


    def validation_step(self, batch: dict, batch_idx: int) -> None:
        raise NotImplementedError


    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError


    def training_epoch_end(self, outputs):
        raise NotImplementedError


    def validation_epoch_end(self, outputs):
        raise NotImplementedError


    def test_epoch_end(self, outputs):
        raise NotImplementedError


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())