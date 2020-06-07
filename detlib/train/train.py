from __future__ import division

import argparse
import datetime
import time

import torch
from terminaltables import AsciiTable
from torch.autograd import Variable

from detlib.detect import train_demo
from detlib.models import *
from detlib.train.test import evaluate
from detlib.utils.datasets import *

from detlib.utils import GradualWarmupScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model related
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--optim", type=str, default="sgd", help="optim")
    # Training related
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--eval_batch_lim", type=int, default=50, help="limit number of batches to test during eval")
    parser.add_argument("--freeze_backbone_until", type=int, default=0, help="freeze backbone for x first epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate value")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # Logger related
    parser.add_argument("--name", type=str, default="", help="run name")
    parser.add_argument("--logger", default=True, help="activate companion Tensorboard istance")
    parser.add_argument("--metric", default=False, help="show metric table?")
    parser.add_argument("--start_epoch", type=int, default=0, help="Active not to mess with logs")
    # ECP related
    parser.add_argument("--ECP", type=int, default=1, help="Using ECP dataset?")
    parser.add_argument("--data", type=str, default="data/", help="Dataset path (if ECP, the folder containing it)")
    parser.add_argument("--town", type=str, default="", help="subset town to train on (ex: to = torin+toulose")


    opt = parser.parse_args()

    if opt.logger:
        # Logger require to instance Tensorflow, we import it only if needed
        logger = Logger("logs/", opt.logger, opt.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # free some memory if occupied

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    data_config = parse_data_config(opt.data_config)

    # Get data configuration
    if not opt.ECP:
        train_path = data_config["train"]
        valid_path = data_config["valid"]
    else:
        train_path = ""
        valid_path = "val"

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
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training, town=opt.town, ecp=opt.ECP)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Define optimized (and scheduler)
    assert opt.optim in ["adam", "sgd"]
    lr = opt.lr
    if opt.optim == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=300, factor=0.33, verbose=True, min_lr=1e-9)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr,
                        weight_decay=0.0001)
        scheduler_RedLR = ReduceLROnPlateau(optimizer, patience=500, factor=0.3, verbose=True, min_lr=1e-8)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_step=500,
                                                  after_scheduler=scheduler_RedLR)

    metrics = [
        "grid_size",
        "loss",
        "conf",
        "recall50",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    log_every = 10  # update tensorboard every 10 steps
    average_steps = 10  # moving average filter for loss function to scheduler
    loss_filtered = -1  # starting value
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        model.set_backbone_grad(epoch >= opt.freeze_backbone_until)
        # logger.scalar_summary("params", model.get_active_params(), epoch) # save # params with active grad

        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            if type(loss) == int:
                continue  # skip when there were no target
            loss.backward()
            loss_filtered = average_filter(loss_filtered, loss.item(), average_steps)

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                if opt.optim == "sgd":
                    scheduler.step(step=batches_done, metrics=loss_filtered)
                    #  clipping exploding gradient to 25
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 25)
                else:
                    scheduler.step(loss_filtered)

                logger.scalar_summary("loss_filtered", loss_filtered, batches_done)
                logger.scalar_summary("lr", optimizer.param_groups[0]['lr'], batches_done)
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
                if batches_done % log_every == 0:  # td change batch_i
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j + 1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += "\n"

            log_str += f"Total loss {loss.item()}\n"
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"Loss_filtered {loss_filtered}\n"\
                       f"Lr - {optimizer.param_groups[0]['lr']}"
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.90,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=16,
                town=opt.town,
                ecp=opt.ECP,
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

            # eval on images in data/samples, store detections in misc/images
            # and load them in tensorboard for visualization
            train_demo(model, logger, epoch_n=epoch, img_size=opt.img_size)

        if epoch % opt.checkpoint_interval == 0:
            print("Saving model")
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_{opt.name}_%d.pth" % epoch)
