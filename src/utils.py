from typing import List
from torch import Tensor
from PIL import Image
from torchvision import transforms
import numpy as np

import os

def join_dirs(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, subdir) for subdir in subdirs]

def paths_images_to_tensor(images_paths: List[str]) -> List[Tensor]:
    images = []
    for image_path in images_paths:
        images.append(transforms.ToTensor()(Image.open(image_path).convert('RGB')))
    return transforms.ToTensor()(np.asarray(images))
        