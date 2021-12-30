import os
import torch

from typing import Generator, List, Optional, Union
from torch.utils.data import DataLoader
from src.data.datasets import RacingF1Dataset
from torchvision.transforms import Compose
from src.utils import join_dirs

import pytorch_lightning as pl

class RacingF1DataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, dataset: RacingF1Dataset) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.dataset = dataset

    @staticmethod
    def from_directories(batch_size: int, dataset_root_dir: str, subdirs: List[str] = None,
                         data_transform: Optional[Compose] = None) -> "RacingF1DataModule":

        if subdirs is None:
            subdirs = os.listdir(dataset_root_dir)
        
        datasets_dirs = join_dirs(dataset_root_dir, subdirs)

        return RacingF1DataModule(batch_size, RacingF1Dataset(datasets_dirs, data_transform))

    def setup(self, stage: Optional[str] = None, train_size: float = 0.6,
              val_size: float = 0.25, test_size: float = 0.15,
              rnd_generator: Generator = torch.Generator().manual_seed(42)) -> None:

        trainset, testset, valset = self.dataset.random_split(train_size, val_size, test_size, rnd_generator)

        if stage == 'fit' or stage is None:
            self.train_dataset = trainset
            self.val_dataset = valset
        
        if stage == 'test' or stage is None:
            self.test_dataset = testset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
