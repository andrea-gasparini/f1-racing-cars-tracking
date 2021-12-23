from collections import Counter
import os
import torch

from torch import Tensor
from typing import List, Optional
from torch._C import Generator, default_generator
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import random_split
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.ops.boxes import box_iou


def join_dirs(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, subdir) for subdir in subdirs]


def tensor_to_image(tensor: Tensor, rgb: bool = True) -> Image.Image:
	return transforms.ToPILImage()(tensor).convert("RGB" if rgb else None)


def image_to_tensor(image: Image.Image) -> Tensor:
	return transforms.ToTensor()(image)


def draw_bounding_box(img: Image.Image, bounding_box: List[int]) -> Image.Image:

	shape = [(bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])]
	image_with_bounding_box = img.copy()
	draw_bounding_box = ImageDraw.Draw(image_with_bounding_box)
	draw_bounding_box.rectangle(shape, outline='red')

	return image_with_bounding_box


def random_split_dataset(dataset: Dataset, train_size: float, test_size: float,
                 		 val_size: Optional[float] = None, generator: Generator = default_generator) -> List[Subset]:

    train_size = int(len(dataset) * train_size)
    test_size = int(len(dataset) * test_size)
    if val_size is not None: val_size = int(len(dataset) * val_size)

    if sum(filter(None, [train_size, test_size, val_size])) < len(dataset):
        train_size += len(dataset) - sum(filter(None, [train_size, test_size, val_size]))

    return random_split(dataset, generator=generator, lengths=[train_size, test_size, val_size])



	
	
