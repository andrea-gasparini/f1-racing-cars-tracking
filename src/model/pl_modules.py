import pytorch_lightning as pl
import torch
import torchmetrics

from typing import Dict, List, Optional, Union
from torch import Tensor
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RacingF1Detector(pl.LightningModule):

    NUM_CLASSES: int = 2
    """
    Number of classes for the FastRCNNPredictor,
    at least 1 for the background and 1 for a type of object
    """

    def __init__(self, hparams = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(hparams)

        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # freeze the weights of the FastRCNN's layers
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.NUM_CLASSES)

        # metrics to track
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1()        
        self.test_f1 = torchmetrics.F1()


    def forward(self, x: Tensor, y: Optional[Tensor] = None,
                **kwargs) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:

        out = self.model(x, y)

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
                target_box = torch.FloatTensor(target_box).unsqueeze(0).to(self.device)
                targets.append({ 'boxes': target_box, 'labels': target_labels })
                inputs = torch.cat([inputs[0:i], inputs[i+1:]])

        out_dict = self.forward(inputs, targets)

        if self.training:
            # out_dict: Dict[str, Tensor]
            loss = sum(loss for loss in out_dict.values())
            return { 'loss': loss }
        else:
            # out_dict: List[Dict[str, Tensor]]
            new_out: Dict[str, Tensor] = dict()

            for out_sample in out_dict:
                for key, value in out_sample.items():

                    unsq_value = value.unsqueeze(0)

                    if key not in new_out: new_out[key] = unsq_value
                    else: new_out[key] = torch.cat((new_out[key], unsq_value))

            boxes = new_out['boxes']
            best_score_idxs = new_out['scores'].argmax(1)

            predictions = [boxes[i].tolist() for boxes, i in zip(boxes, best_score_idxs)]
            new_out['predictions'] = torch.tensor(predictions)

            return new_out


    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        out = self.step(batch)

        loss = out['loss']

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        out = self.step(batch)

        predictions = out['predictions']
        labels = batch['bounding_box']

        accuracy = self.val_acc(predictions, labels)
        f1_score = self.val_f1(predictions, labels)

        self.log_dict({
                'val_acc': accuracy,
                'val_f1': f1_score,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )


    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        out = self.step(batch)

        predictions = out['predictions']
        labels = batch['bounding_box']

        accuracy = self.test_acc(predictions, labels)
        f1_score = self.test_f1(predictions, labels)

        self.log_dict({
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

        params = [p for p in self.model.parameters() if p.requires_grad]

        return torch.optim.Adam(params)