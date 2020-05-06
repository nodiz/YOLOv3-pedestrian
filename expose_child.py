from __future__ import division

import argparse
import datetime
import os
import time

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader

from detect_train import demo
from models import *
from test import evaluate
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--logger", default=True, help="activate companion Tensorboard istance (BUGGED!)")
    parser.add_argument("--town", type=str, default="", help="subset town to train on")
    parser.add_argument("--overfit", default=False, help="eval on train?")
    parser.add_argument("--metric", default=False, help="show metric table?")
    parser.add_argument("--eval_batch_lim", type=int, default=50, help="number of batches to test on during eval")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate value")
    parser.add_argument("--name", type=str, default="", help="run name")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # free some elves!

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is -")
        print(child)
        child_counter += 1



