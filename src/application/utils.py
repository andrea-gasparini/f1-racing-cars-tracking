import cv2
import os
import torchvision
import json
import matplotlib.pyplot as plt

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

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    
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
    pbar = tqdm(sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))))
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


def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction


def remove_low_scores(orig_prediction, score_thresh=0.2):
  final_prediction = {"boxes": [], "scores": [], "labels": []}
  for idx, item in enumerate(orig_prediction["scores"]):
    if item > score_thresh:
      final_prediction["boxes"].append(orig_prediction["boxes"][idx])
      final_prediction["scores"].append(item)
      final_prediction["labels"].append(1)
  
  return final_prediction


def generate_bounding_boxes(model_ckpt_path: str, frames_path: str, size_dirs: int = 5) -> None:
    model = RacingF1Detector.load_from_checkpoint(model_ckpt_path)
    model.eval()
    
    images_dir = sorted(f for f in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, f)))

    output_boxes = {}
    for images_dir_chunk in chunks(images_dir, size_dirs):
        images = []

        pbar = tqdm(images_dir_chunk)
        for image_name in pbar:
            image_full_path = os.path.join(frames_path, image_name)
            pbar.set_description(f"Reading {image_name}")
            images.append(image_to_tensor(Image.open(image_full_path)))
            

        print("Computing predictions...")
        outputs = model(images)

        output_path = os.path.join(frames_path, 'bounding_box')

        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        pbar = tqdm(enumerate(images_dir_chunk))
        for idx, image_name in pbar:
            image_full_path = os.path.join(frames_path, image_name)
            pbar.set_description(f"Drawing bounding box on {image_name}")
            pred = remove_low_scores(outputs[idx])
            pred = apply_nms(outputs[idx])
            img = draw_bounding_box(Image.open(image_full_path), pred["boxes"])
            output_boxes[image_name] = {"scores": pred["scores"], "boxes": pred["boxes"]}
            img.save(os.path.join(output_path, image_name), "JPEG")
                    
        images = []
    return output_boxes
    
    
def calculate_histogram_bounding_box(image_path: str, bbox_file_path: str):
    image = cv2.imread(image_path)
    
    with open(bbox_file_path, mode='r') as f:
        bounding_boxes = json.load(f)
    
    histograms = []
    for bbox in bounding_boxes:
        bbox = [int(i) for i in bbox]
        crop_img = image[bbox[1]+2:bbox[3]-2, bbox[0]+2:bbox[2]-2]
        hist = cv2.calcHist([crop_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        histograms.append(hist)
    return histograms
    
def calculate_histogram_distance(hist1, hist2, method=cv2.HISTCMP_INTERSECT):
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    
    return cv2.compareHist(hist1, hist2, method)

