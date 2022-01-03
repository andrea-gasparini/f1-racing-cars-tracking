import cv2
import os

from tqdm import tqdm

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.datasets import RacingF1Dataset
from src.model.pl_modules import RacingF1Detector
from src.utils import image_to_tensor, draw_bounding_box
from PIL import Image

def split_video_to_frames(path: str, output_path: str, zfill: int = 4) -> None:

    assert path != '' and output_path != ''

    if not os.path.isfile(path):
        raise ValueError(f"Expected {path} to be a full path to a video file")
    
    video_capture = cv2.VideoCapture(path)
    success, image = video_capture.read()
    count = 0
    
    path = os.path.join(path)
    
    while success:
        cv2.imwrite(os.path.join(output_path, f"frame_{str(count).zfill(zfill)}.jpg"), image)
        success, image = video_capture.read()
        count += 1
    
    print(f"Done {str(count)} frames.")


def merge_frames_to_video(path: str, out_path: str, fps: int = 60) -> None:

    img_array = list()

    # Read frames from directory
    pbar = tqdm(sorted(os.listdir(os.path.join(path))))
    for filename in pbar:
        pbar.set_description(f"Reading {filename}")
        img = cv2.imread(os.path.join(path, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "out_video.avi")
    
    # Merge frames and write them into a video
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    
    pbar = tqdm(range(len(img_array)))
    for i in pbar:
        pbar.set_description(f"Writing frame #{i}")
        out.write(img_array[i])

    out.release()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_bounding_boxes(model_ckpt_path: str, frames_path: str, size_dirs: int = 5) -> None:
    model = RacingF1Detector.load_from_checkpoint(model_ckpt_path)
    model.eval()
    
    images_dir = sorted(f for f in os.listdir(os.path.join(frames_path)) if os.path.isfile(os.path.join(frames_path, f)))
    
    #print(images_dir)
    for images_dir_chunk in chunks(images_dir, size_dirs):
        images = []

        pbar = tqdm(images_dir_chunk)
        for image_name in pbar:
            image_full_path = os.path.join(frames_path, image_name)
            if os.path.isfile(image_full_path):
                pbar.set_description(f"Reading {image_name}")
                images.append(image_to_tensor(Image.open(image_full_path)))
            

        print("Computing predictions...")
        outputs = model(images)
        
        pbar = tqdm(enumerate(images_dir_chunk))
        for idx, image_name in pbar:
            image_full_path = os.path.join(frames_path, image_name)
            if os.path.isfile(image_full_path):
                pbar.set_description(f"Drawing bounding box on {image_name}")
                #print(idx)
                img = draw_bounding_box(Image.open(image_full_path), outputs[idx]['boxes'][0])
                img.save(os.path.join(frames_path, 'bounding_box/') + image_name, "JPEG")
                    
        images = []
    
    
    
    
    