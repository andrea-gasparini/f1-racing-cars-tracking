from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union

import os

class RacingF1Dataset(Dataset):

    IMG_SUBDIR = 'img'
    GROUNDTRUTH_FILENAME = 'groundtruth.txt'

    def __init__(self, dataset_dirs: List[str]) -> None:
        self.dataset_dirs: List[str] = dataset_dirs
        self.samples: List[dict] = list()

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
                        imgs_list[idx]['bounding_box'] = { 'x': int(groundtruth_values[0]),
                                                           'y': int(groundtruth_values[1]),
                                                           'width': int(groundtruth_values[2]),
                                                           'height': int(groundtruth_values[3]) }

            
            self.samples += imgs_list


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, Union[str, Optional[Dict[str, int]]]]:
        return self.samples[idx]


if __name__ == '__main__':
    a = RacingF1Dataset(["dataset/racing-1"])
    for x in [a[0], a[1], a[200]]:
        print(x)