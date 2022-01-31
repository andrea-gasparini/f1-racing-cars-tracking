import cv2
import os
import torchvision
import json
import matplotlib.pyplot as plt
import math

from tqdm import tqdm

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.datasets import RacingF1Dataset
from src.model.pl_modules import RacingF1Detector
from src.utils import image_to_tensor, draw_bounding_box
from PIL import Image
from typing import List

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


def apply_nms(orig_prediction, iou_thresh: float):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction


def remove_low_scores(orig_prediction, score_thresh: float):
  final_prediction = {"boxes": [], "scores": [], "labels": []}
  for idx, item in enumerate(orig_prediction["scores"]):
    if item > score_thresh:
      final_prediction["boxes"].append(orig_prediction["boxes"][idx])
      final_prediction["scores"].append(item)
      final_prediction["labels"].append(1)
  
  return final_prediction


def generate_bounding_boxes(model_ckpt_path: str, frames_path: str, size_dirs: int = 5,
                            score_thresh: float = 0.2, iou_thresh: float = 0.5) -> None:
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
        output_path_bbox = os.path.join(frames_path, 'txt_bounding_box')

        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        if not os.path.isdir(output_path_bbox):
            os.mkdir(output_path_bbox)

        pbar = tqdm(enumerate(images_dir_chunk))
        for idx, image_name in pbar:
            image_full_path = os.path.join(frames_path, image_name)
            pbar.set_description(f"Drawing bounding box on {image_name}")
            pred = remove_low_scores(outputs[idx], score_thresh)
            pred = apply_nms(outputs[idx], iou_thresh)
            with open(os.path.join(frames_path, 'txt_bounding_box', image_name.replace('.jpg', '.txt')), mode='w') as f:
              json.dump(pred["boxes"].detach().numpy().tolist(), f)
            img = draw_bounding_box(Image.open(image_full_path), pred["boxes"])
            img.save(os.path.join(output_path, image_name), "JPEG")     
        images = []
    #return output_boxes
    
    
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
    
    

def calculate_histogram_from_coordinates(image_path: str, coordinates: List[int]):
    image = cv2.imread(image_path)
    histograms = []
    coordinates = [int(i) for i in coordinates]
    #for coord in coordinates:
    crop_img = image[coordinates[1]+2:coordinates[3]-2, coordinates[0]+2:coordinates[2]-2]
    hist = cv2.calcHist([crop_img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    histograms.append(hist)
    return histograms

def calculate_histogram_distance(hist1, hist2, method=cv2.HISTCMP_INTERSECT):
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    
    return cv2.compareHist(hist1, hist2, method)



def check_bounding_boxes(frames_path: str):
    txt_path = os.path.join(frames_path, 'txt_bounding_box')
    images_path = os.path.join(frames_path, 'bounding_box')
    
    images = os.listdir(images_path)
    for i, image_name in enumerate(images):
        
        if i == len(images) -1:
            # we do not need to check the last element.
            break
        
        
        first_image = os.path.join(images_path, image_name)
        second_image = os.path.join(images_path, images[i + 1])
        
        with open(os.path.join(txt_path, image_name.replace('.jpg', '.txt')), mode='r') as f:
            first_image_bbox = json.load(f)
        
        with open(os.path.join(txt_path, images[i + 1].replace('.jpg', '.txt')), mode='r') as f:
            second_image_bbox = json.load(f)
            

        if len(first_image_bbox) > len(second_image_bbox):
            pass
        elif len(first_image_bbox) < len(second_image_bbox):
            pass
        else:
            equal_bboxes_length(first_image_bbox, second_image_bbox, image_name, images[i + 1])
            
        
        print(first_image_bbox, second_image_bbox)
        break


def calculate_centroids(x1, x2, y1, y2):
    return((x1 + x2)/2, (y1 + y2)/2)


def first_greater_second(first_bboxes: List[float], second_bboxes: List[float], first_image_name: str, second_image_name: str):
    # case in which the prev frame has > bboxes than the second.
    tmp_first_bboxes = first_bboxes
    for first_bbox in first_bboxes:
        for second_bbox in second_bboxes:
            first_centroid = calculate_centroids(first_bbox[0], first_bbox[2], first_bbox[1], first_bbox[3])
            second_centroid = calculate_centroids(second_bbox[0], second_bbox[2], second_bbox[1], second_bbox[3])
            
            distance = math.hypot(second_centroid[0] - first_centroid[0], second_centroid[1] - first_centroid[1])
            if distance < 50: # threshold
                tmp_first_bboxes.remove(first_bbox) # same bbox
                break
            
    # now in tmp_first_bboxes we have only boxes to check with histograms.
            
            
    

def equal_bboxes_length(first_bboxes: List[float], second_bboxes: List[float], first_image_name: str, second_image_name: str):
    for idx, bbox in enumerate(first_bboxes):
        first_centroid = calculate_centroids(bbox[0], bbox[2], bbox[1], bbox[3])
        second_centroid = calculate_centroids(second_bboxes[idx][0], second_bboxes[idx][2], second_bboxes[idx][1], second_bboxes[idx][3])
        
        distance = math.hypot(second_centroid[0] - first_centroid[0], second_centroid[1] - first_centroid[1])
        
        if distance > 50:
            # they are, probably, two different bounding boxes and we need to check better
            pass