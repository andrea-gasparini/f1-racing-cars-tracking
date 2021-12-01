import pytorch_lightning as pl
import torchvision.models as models
import torch

from typing import Any, List
from torch import Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn import Module, Sequential, Linear, Dropout, ReLU, Sigmoid, MSELoss

from src.utils import paths_images_to_tensor


class RacingF1Detector(pl.LightningModule):

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)
        self.loss_function = MSELoss()

        vgg16 = models.vgg16(pretrained=True)
        vgg16.training = False
        
        for param in vgg16.features.parameters():
            param.requires_grad = False
            
        vgg_layers: List[Module] = list(vgg16.classifier.children())
        vgg_last_linear_layer: Linear = vgg_layers[-1]
        vgg_out_size: int = vgg_last_linear_layer.out_features

        head_layers = Sequential(
            *vgg_layers,
            Dropout(0.5),
            Linear(vgg_out_size, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )

        self.model = torch.nn.Sequential(head_layers)


    def forward(self, x: List[Tensor], **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        out = self.model(x)
        return out


    def training_step(self, batch: dict, batch_idx: int) -> Tensor:

        inputs = paths_images_to_tensor(batch['img_path'])
        labels = batch['bounding_box']

        logits = self.forward(inputs)

        loss = self.loss_function(logits[0], labels['x'])
        loss += self.loss_function(logits[1], labels['y'])
        loss += self.loss_function(logits[2], labels['width'])
        loss += self.loss_function(logits[3], labels['height'])

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