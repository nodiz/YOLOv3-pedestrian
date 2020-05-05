import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import json


classes_dict = {
    "pedestrian": 0,
    "rider": 1,
    "person-group-far-away": 2,
}

class bbox_rect(object):
    def __init__(self, s):
        self.x0, self.x1, self.y0, self.y1 = s['x0'], s['x1'], s['y0'], s['y1']
        self.xc = (self.x0 + self.x1) / 2
        self.yc = (self.y0 + self.y1) / 2
        self.w = abs(self.x1 - self.x0)
        self.h = abs(self.y1 - self.y0)
        self.h0 = 1024
        self.w0 = 1920

    def to_ecp(self):
        return [self.x0, self.x1, self.y0, self.y1]

    def to_yolo(self):
        return [self.xc, self.yc, self.w, self.h]

    def to_yolo_norm(self):
        l_yolo = self.to_yolo()
        l_yolo[0] /= self.w0
        l_yolo[1] /= self.h0
        l_yolo[2] /= self.w0
        l_yolo[3] /= self.h0
        return l_yolo


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/day/img/train/*/*.*" % folder_path))  # ECP/day/img/train/wuerzburg/wuerzburg_00673.png
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, folder_path, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                 ECP_PATH="/home/nodiz/dlav_project/data/ECP", town=""):
        #with open(list_path, "r") as file:
        #    self.img_files = file.readlines()
        assert folder_path in ["train", "val"]

        self.img_files = sorted(glob.glob("{}/day/img/{}/{}*/*.*".format(ECP_PATH,folder_path, town)))

        self.label_files = [
            path.replace("img", "labels").replace(".png", ".json").replace(".jpg", ".json")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):

            with open(label_path, 'r') as f:
                label_data = json.load(f)['children']

                list_of_bbox = []

                for bbox in label_data:
                    if bbox['identity'] in classes_dict.keys():
                        bbox_id = classes_dict[bbox['identity']]
                        bbox = bbox_rect(bbox).to_yolo_norm()
                        list_of_bbox.append([bbox_id] + bbox)
                if len(list_of_bbox) == 0:
                    print("found zero label data: {}".format(label_path))
                list_of_bbox = np.array(list_of_bbox)

            boxes = torch.from_numpy(list_of_bbox.reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
