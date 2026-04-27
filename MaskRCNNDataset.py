import os
import torch

from torchvision.io import read_image, decode_image
from PIL import Image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset

from glob import glob
import json
import numpy as np
import cv2

class MyDataset(Dataset):
    def __init__(self, dataset_dir, transforms):
        self.transforms = transforms
        self.dataset_dir = dataset_dir
        self.json_paths = glob(os.path.join(dataset_dir, "*.json"))
        self.class_ids = {"tab": 1, "bead": 2}

    def __len__(self):
        return len(self.json_paths)
    
    def __getitem__(self, idx):
        # load images and masks
        json_path = self.json_paths[idx]
        with open(json_path, 'r') as jsonfile:
            jsondata = json.load(jsonfile)
        image_path = os.path.join(self.dataset_dir, jsondata['imagePath'])
        img = Image.open(image_path).convert("RGB") 
        masks, labels, boxes, area, iscrowd = self.get_instances(jsondata)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = torch.tensor(area)
        target["iscrowd"] = torch.tensor(iscrowd)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def get_instances(self, jsondata):
        height = jsondata["imageHeight"]
        width = jsondata["imageWidth"]
        shapes = jsondata["shapes"]
        masks = []
        labels = []
        boxes = []
        area = []
        iscrowd = []
        for shape in shapes:
            mask = np.zeros((height, width), dtype=np.uint8)
            label = shape["label"]
            points = np.array(shape["points"], dtype=int)
            mask = cv2.fillPoly(mask, [points], 1)
            masks.append(mask)                  # [mask(540,720), mask(540,720), mask(540,720), ..., mask(540,720)] : shape=(10, 540, 720)
            labels.append(self.class_ids[label])
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            area.append((xmax-xmin)*(ymax-ymin))
            iscrowd.append(0)
        masks = np.array(masks)
        labels = np.array(labels)
        boxes = np.array(boxes)
        area = np.array(area, dtype=np.float32)
        iscrowd = np.array(iscrowd)
        return masks, labels, boxes, area, iscrowd
    
    

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)