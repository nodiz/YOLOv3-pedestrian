from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


def det2json(detections):
    mock_detections = []
    return # tbd
    for box in detections[0][0]:
        box = {'x0': float(box[0]),
               'x1': float(box[2]),
               'y0': float(box[1]),
               'y1': float(box[3]),
               'score': float(box[4]),
               'identity': 'pedestrian',
               'orient': 0.0}
        mock_detections.append(box)
    return mock_detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--destdir", type=str, default="output_json/", help="Dataset path (if ECP, the folder containing it)")

    # ECP related
    parser.add_argument("--dataset", type=str, default="ECP", help="ECP or none")
    parser.add_argument("--data", type=str, default="data/", help="Dataset path (if ECP, the folder containing it)")
    parser.add_argument("--town", type=str, default="", help="subset town to train on (ex: to = torin+toulose")

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    destdir = opt.destdir
    os.makedirs(destdir, exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.data, img_size=opt.img_size,
                    dataset=opt.dataset, folder_scope="val", folder_town=opt.town),  # to correct like in other loader
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(tqdm.tqdm(dataloader)):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

        print("\nSaving json:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            # Rescale boxes to original image
            if detections is not None:

                detections = rescale_boxes(detections, opt.img_size, (1024, 1920))
                detections_json = []
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box = {'x0': float(x1),
                           'x1': float(x2),
                           'y0': float(y1),
                           'y1': float(y2),
                           'score': float(cls_conf.item()),
                           'identity': 'pedestrian',
                           'orient': 0.0}
                    detections_json.append(box)

            # create json
            destfile = os.path.join(destdir, os.path.basename(path).replace('.png', '.json'))
            frame = {'identity': 'frame',
                     'children': detections_json}
            json.dump(frame, open(destfile, 'w'), indent=1)

        imgs = []  # reset for next batch
        img_detections = []