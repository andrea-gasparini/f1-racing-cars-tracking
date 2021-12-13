import pytorch_lightning as pl
import torch
import torchvision

from typing import Any, List
from torch import Tensor
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import MSELoss
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models as models

class RacingF1Detector(pl.LightningModule):

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)
        self.loss_function = MSELoss()

        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = Sequential(*layers)
        self.classifier = Linear(num_filters, 4)

    def forward(self, x: List[Tensor], **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x


    def training_step(self, batch: dict, batch_idx: int) -> Tensor:

        inputs = batch['img']
        labels = torch.Tensor([label.tolist() for label in batch['bounding_box']]).cuda()
        labels = torch.einsum('ij->ji', labels)
        logits = self.forward(inputs)
        loss = self.loss_function(logits, labels)
        
        self.log("Train loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch: dict, batch_idx: int) -> None:
        raise NotImplementedError


    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError


    def training_epoch_end(self, outputs):
      ...


    def validation_epoch_end(self, outputs):
        raise NotImplementedError


    def test_epoch_end(self, outputs):
        raise NotImplementedError


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())