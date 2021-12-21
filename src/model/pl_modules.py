import pytorch_lightning as pl
import torch
import torchmetrics

from typing import Dict, List, Union
from torch import Tensor
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RacingF1Detector(pl.LightningModule):

    OUT_VALUES: int = 4

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.OUT_VALUES)

        # metrics to track
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1()        
        self.test_f1 = torchmetrics.F1()


    def forward(self, x: Tensor, **kwargs) -> Tensor:

        out = self.model(x)

        return out

    def step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:

        inputs = batch['img']
        labels = batch['bounding_box']

        target_labels = torch.tensor([1], device=self.device)

        targets = list()
        for i, label in enumerate(labels):
            target_box = [label[0], label[1] - label[3], label[0] + label[2], label[1]]
            if target_box[0] == target_box[2] or target_box[1] == target_box[3]:
                continue
            else:
                target_box = torch.FloatTensor(target_box).reshape(1, self.OUT_VALUES).to(self.device)
                targets.append({ 'boxes': target_box, 'labels': target_labels })
                inputs = torch.cat([inputs[0:i], inputs[i+1:]])

        out_dict: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = self.model(inputs, targets)

        if self.training:
            # out_dict: Dict[str, Tensor]
            loss = sum(loss for loss in out_dict.values())
            return { 'loss': loss }
        else:
            # out_dict: List[Dict[str, Tensor]]
            new_out_dict = dict()

            for out_sample in out_dict:
                for key, value in out_sample.items():

                    tensor_value = torch.tensor([value])

                    if key not in new_out_dict: new_out_dict[key] = tensor_value
                    else: new_out_dict[key] = torch.cat((new_out_dict[key], tensor_value))

            return new_out_dict


    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        out = self.step(batch)

        loss = out['loss']

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        out = self.step(batch)

        predictions = out['boxes']
        labels = batch['labels']

        accuracy = self.val_acc(predictions, labels)
        f1_score = self.val_f1(predictions, labels)

        self.log_dict(
            {
                'val_acc': accuracy,
                'val_f1': f1_score,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )


    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        out = self.step(batch)

        predictions = out['boxes']
        labels = batch['labels']

        accuracy = self.test_acc(predictions, labels)
        f1_score = self.test_f1(predictions, labels)

        self.log_dict(
            {
                'test_acc': accuracy,
                'test_f1': f1_score,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )


    def training_epoch_end(self, outputs):
        pass


    def validation_epoch_end(self, outputs):
        raise NotImplementedError


    def test_epoch_end(self, outputs):
        raise NotImplementedError


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())