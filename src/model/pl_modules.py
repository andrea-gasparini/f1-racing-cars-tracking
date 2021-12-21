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
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RacingF1Detector(pl.LightningModule):

    OUT_VALUES: int = 4

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)
        self.loss_function = MSELoss()
        
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.OUT_VALUES)


    def forward(self, x: Tensor, **kwargs) -> dict:

        out = self.model(x)

        return out


    def training_step(self, batch: dict, batch_idx: int) -> Tensor:

        inputs = batch['img']
        labels = torch.tensor([label.tolist() for label in batch['bounding_box']])
        labels = torch.einsum('ij -> ji', labels)
 
        target_labels = torch.tensor([1])
 
        targets = list()
        for i, label in enumerate(labels):
            target_box = [label[0], label[1] - label[3], label[0] + label[2], label[1]]
            if target_box[0] == target_box[2] or target_box[1] == target_box[3]:
              continue
            else:
              target_box = torch.FloatTensor(target_box).reshape(1, self.OUT_VALUES)
              targets.append({ 'boxes': target_box, 'labels': target_labels })
              inputs = torch.cat([inputs[0:i], inputs[i+1:]])
 
        
        loss_dict = self.model(inputs, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch: dict, batch_idx: int) -> None:
        raise NotImplementedError


    def test_step(self, batch: dict, batch_idx: int) -> Any:
        raise NotImplementedError


    def training_epoch_end(self, outputs):
        pass


    def validation_epoch_end(self, outputs):
        raise NotImplementedError


    def test_epoch_end(self, outputs):
        raise NotImplementedError


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())