import os

from typing import List
from PIL import Image, ImageDraw


def join_dirs(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, subdir) for subdir in subdirs]


def draw_bounding_box(img: Image.Image, bounding_box: List[int]) -> Image.Image:

	shape = [(bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])]
	image_with_bounding_box = img.copy()
	draw_bounding_box = ImageDraw.Draw(image_with_bounding_box)
	draw_bounding_box.rectangle(shape, outline='red')

	return image_with_bounding_box