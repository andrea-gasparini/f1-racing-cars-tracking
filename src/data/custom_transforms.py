from typing import Dict, List, Tuple, Union
from PIL import Image
from torchvision.transforms import functional as TF

class CustomRescale(object):

    def __init__(self, output_size: Tuple[int, int]):
        self.output_size = output_size

    
    def __call__(self, sample: Tuple[Image.Image, List[int]]) -> Dict[str, Union[Image.Image, List[int]]]:
        image, bounding_box = sample[0], sample[1]

        h, w = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = TF.resize(image, (new_h, new_w))

        b_box = [int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])]
        x_new = int(b_box[0]/(image.width/new_w))
        y_new = int(b_box[1]/(image.height/new_h))
        # Example of re-scaling [291, 63, 32, 13] -> [68, 25, ?, ?]
        # 1. 291 : 32 = 68 : x -> 32 * 68 / 291
        w_new = int(b_box[2] * x_new / b_box[0])
        h_new = int(b_box[3] * y_new / b_box[1])

        return { 'img': img, 'bounding_box': [x_new, y_new, w_new, h_new] }