import os
import torch

from torch import Tensor
from typing import List, Optional, Dict
from torch._C import Generator, default_generator
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import random_split
from PIL import Image, ImageDraw
from torchvision import transforms


def join_dirs(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, subdir) for subdir in subdirs]


def tensor_to_image(tensor: Tensor, rgb: bool = True) -> Image.Image:
	return transforms.ToPILImage()(tensor).convert("RGB" if rgb else None)


def image_to_tensor(image: Image.Image) -> Tensor:
	return transforms.ToTensor()(image)


def draw_bounding_box(img: Image.Image, bounding_box: List[List[int]]) -> Image.Image:

  image_with_bounding_box = img.copy()
  for bbox in bounding_box:
    image_with_bounding_box = image_with_bounding_box.copy()
    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
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


def collate_fasterrcnn_predictions(out_dict: List[Dict[str, Tensor]]):
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


	
	
