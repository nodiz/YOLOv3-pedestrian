from __future__ import division

import argparse
import datetime
import os
import time

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from torch.optim.sgd import SGD

from detect_train import demo
from models import *
from test import evaluate
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

from utils.scheduler import GradualWarmupScheduler
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
    parser.add_argument("--logger", default=True, help="activate companion Tensorboard istance")
    parser.add_argument("--town", type=str, default="", help="subset town to train on")
    parser.add_argument("--overfit", default=False, help="eval on train?")
    parser.add_argument("--metric", default=False, help="show metric table?")
    parser.add_argument("--eval_batch_lim", type=int, default=50, help="number of batches to test on during eval")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate value")
    parser.add_argument("--name", type=str, default="", help="run name")
    parser.add_argument("--start_epoch", type=int, default=0, help="not done training?")
    parser.add_argument("--freeze_backbone_until", type=int, default=0, help="freeze backbone for x first steps")

    opt = parser.parse_args()

    if opt.logger:
        from utils.logger_torch import *
        logger = Logger("bck_check/tensorboard/", opt.logger, opt.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # free some elves!

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # train_path = data_config["train"]
    # valid_path = data_config["valid"]
    # for ECP
    train_path = "train"
    if opt.overfit:
        valid_path = "train"
    else:
        valid_path = "val"

    town = opt.town

    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=not opt.overfit, multiscale=opt.multiscale_training, town=town)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, momentum=0.9, weight_decay=0.0001)
    
    scheduler_RedLR = ReduceLROnPlateau(optimizer, patience=500, factor=0.3, verbose=True, min_lr=1e-8)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_step=3000, after_scheduler=scheduler_RedLR)

    # filter(lambda p: p.requires_grad, model.parameters())
    metrics = [
        "grid_size",
        "loss",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    log_every = 10
    t_steps = 0
    average_steps = 30
    loss_filtered = -1
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        model.set_backbone_grad(epoch >= opt.freeze_backbone_until)

        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            if type(loss) == int:
                continue
            loss.backward()
            # loss_filtered = average_filter(loss_filtered, loss.item(),average_steps)

            loss_filtered = average_filter(loss_filtered, loss.item(),average_steps)

            if opt.logger:
                logger.scalar_summary("loss_filtered", loss_filtered, batches_done)
                logger.scalar_summary("lr", optimizer.param_groups[0]['lr'], batches_done)
            
            if batches_done % opt.gradient_accumulations==0:
                # Accumulates gradient before each step
                print(f"train: step is {batches_done}")
                print(f"loss: step is {loss_filtered}")
                scheduler_warmup.step(step=batches_done, metrics=loss_filtered)
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            if opt.metric:
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                if t_steps % log_every == 0:  # td change batch_i
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j + 1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += "\n"

            log_str += f"Total loss {loss.item()}"
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            t_steps += 1

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=24,
                town=town,
                batch_lim=opt.eval_batch_lim
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP

            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            print("Running demo")
            demo(model, logger, epoch_n=epoch, img_size=opt.img_size)

        if epoch % opt.checkpoint_interval == 0:
            print("saving model")
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_{opt.name}_%d.pth" % epoch)
