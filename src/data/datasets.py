from torch.utils.data import Dataset
from torch import Tensor
from typing import List, Dict, Optional, Union
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np

import os

class RacingF1Dataset(Dataset):

    IMG_SUBDIR = 'img'
    GROUNDTRUTH_FILENAME = 'groundtruth.txt'

    def __init__(self, dataset_dirs: List[str], transforms = None) -> None:
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
                        if not self.transforms:
                            imgs_list[idx]['bounding_box'] = [int(groundtruth_values[0]), int(groundtruth_values[1]), int(groundtruth_values[2]), int(groundtruth_values[3])]
                        else:
                            img = Image.open(imgs_list[idx]['img_path'])
                            w, h = self.transforms[0].size
                            b_box = [int(groundtruth_values[0]), int(groundtruth_values[1]), int(groundtruth_values[2]), int(groundtruth_values[3])]
                            x_new = int(b_box[0]/(img.width/w))
                            y_new = int(b_box[1]/(img.height/h))
                            # Example of re-scaling [291, 63, 32, 13] -> [68, 25, ?, ?]
                            # 1. 291 : 32 = 68 : x -> 32 * 68 / 291
                            w_new = int(b_box[2] * x_new / b_box[0])
                            h_new = int(b_box[3] * y_new / b_box[1])
                            imgs_list[idx]['bounding_box'] = [x_new, y_new, w_new, h_new]
            self.samples += imgs_list


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, Optional[Dict[str, int]]]]:
        sample_img_path = self.samples[idx]['img_path']
        img = Image.open(sample_img_path)
        convert_tensor = transforms.ToTensor()
        
        if self.transforms:
            # convention: first transform is the resize!
            img = self.transforms(img)

        return {
			'img': convert_tensor(img),
			'bounding_box': self.samples[idx]['bounding_box']
		}

    def show_image(self, idx: int) -> None:
        image = Image.open(self.samples[idx]['img_path'])
        if self.transforms:
            image = self.transforms(image)
        bounding_box = self.samples[idx]['bounding_box']
        shape = [(bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])]
        image_copy = image.copy()
        draw_bounding_box = ImageDraw.Draw(image_copy)
        draw_bounding_box.rectangle(shape, outline='red')
        image_copy.save('tesporaposdp.png')
        return image, image_copy
