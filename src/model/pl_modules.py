import pytorch_lightning as pl
import torch
import torchmetrics

from typing import Dict, List, Optional, Union, Tuple
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
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.NUM_CLASSES)

        # metrics to track
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#map
        self.test_mAP = torchmetrics.MAP()
        self.val_mAP = torchmetrics.MAP()
        
    def forward(self, x: Tensor, y: Optional[Tensor] = None,
                **kwargs) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        out = self.model(x, y)
        return out

    def step(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        inputs = batch['img']
        labels = batch['bounding_box']

        target_labels = torch.tensor([1], device=self.device)

        targets = list()
        for i, label in enumerate(labels):

            bbox_is_invalid = label[0] == label[2] or label[1] == label[3] # bounding box area is 0
            
            if bbox_is_invalid:
                # remove sample from inputs tensor
                inputs = torch.cat([inputs[0:i], inputs[i+1:]])
            else:
                bbox = label.unsqueeze(0).to(self.device)
                targets.append({ 'boxes': bbox, 'labels': target_labels })

        out_dict = self.forward(inputs, targets)

        if self.training:
            # out_dict: Dict[str, Tensor]
            loss = sum(loss for loss in out_dict.values())
            return { 'loss': loss }
        
        # out_dict: List[Dict[str, Tensor]]
        return out_dict, targets

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        out = self.step(batch)

        loss = out['loss']

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        
        out, targets = self.step(batch)

        self.val_mAP.update(out, targets)
        result = self.val_mAP.compute()

        self.log('val_mAP', result['map'], logger=True, prog_bar=True)
        self.log('val_mAR', result['mar'], logger=True, prog_bar=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        out, targets = self.step(batch)
        
        self.test_mAP.update(out, targets)
        result = self.test_mAP.compute()

        self.log('test_mAP', result['map'], logger=True, prog_bar=True)
        self.log('test_mAR', result['mar'], logger=True, prog_bar=True)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=self.hparams["lr"],
									   momentum=self.hparams['momentum'],
									   weight_decay=self.hparams['weight_decay'])