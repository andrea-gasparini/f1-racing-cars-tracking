from torch._C import default_generator, Generator
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from PIL import Image
from src.utils import image_to_tensor, random_split_dataset
from torchvision.transforms import Compose
from torch import Tensor

import torch
import os

class RacingF1Dataset(Dataset):

    IMG_SUBDIR = 'img'
    GROUNDTRUTH_FILENAME = 'groundtruth.txt'

    def __init__(self, dataset_dirs: List[str], transforms: Optional[Compose] = None) -> None:

        self.dataset_dirs: List[str] = dataset_dirs
        self.samples: List[dict] = list()
        self.transforms = transforms

        for dir in self.dataset_dirs:			
            groundtruth_dir: str = os.path.join(dir, self.GROUNDTRUTH_FILENAME)
            imgs_dir: str = os.path.join(dir, self.IMG_SUBDIR)
            imgs_dirs: List[str] = sorted(os.listdir(imgs_dir))
            imgs_list = [{'img_path': os.path.join(imgs_dir, subdir)} for subdir in imgs_dirs]
            if os.path.isfile(groundtruth_dir):
                with open(groundtruth_dir, mode='r') as f:
                    groundtruth_list = f.readlines()
                    for idx, value in enumerate(groundtruth_list):
                        groundtruth_values = value.strip().split(',')
                        imgs_list[idx]['bounding_box'] = [int(groundtruth_values[0]), int(groundtruth_values[1]), int(groundtruth_values[2]), int(groundtruth_values[3])]
            self.samples += imgs_list

        
    def random_split(self, train_size: float, test_size: float, val_size: Optional[float] = None,
                    generator: Generator = default_generator) -> List['RacingF1Dataset']:

        return random_split_dataset(self, train_size, test_size, val_size, generator)

    
    def __preprocess_sample(self, idx: int) -> Dict[str, Tensor]:
        sample_bounding_box: List[int] = self.samples[idx]['bounding_box']
        img: Image.Image = Image.open(self.samples[idx]['img_path'])

        # map [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax] convention
        # where (0,0) is the bottom-left corner
        sample_bounding_box = [
            sample_bounding_box[0],
            sample_bounding_box[1],
            sample_bounding_box[0] + sample_bounding_box[2],
            sample_bounding_box[1] + sample_bounding_box[3]
        ]
        
        if self.transforms:
            transformed_sample = self.transforms((img, sample_bounding_box))
            img = transformed_sample['img']
            sample_bounding_box = transformed_sample['bounding_box']

        return {
            'img': image_to_tensor(img),
            'bounding_box': torch.tensor(sample_bounding_box)
        }


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        preprocessed_sample = self.__preprocess_sample(idx)

        return {
            'img': preprocessed_sample['img'],
            'bounding_box': preprocessed_sample['bounding_box']
        }
