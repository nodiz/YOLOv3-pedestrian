# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation adapted for Pedestrian detection and made compatible with the ECP Dataset (https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection).

## Installation
##### Clone and install requirements

Requirement are detailed in requirement.txt

This repository is mantained for:
* Python: 3.7
* Pytorch: 1.4
* CUDA: 10.1

##### Download pretrained weights

Pretrained weights and our model can be found there: 
https://drive.google.com/drive/folders/1DRPNNJoIbM7utW-kDCdCFr7m4gZ0BVvo?usp=sharing

##### Download ECP 
    
Download ECP dataset : https://eurocity-dataset.tudelft.nl after creating an account    
## Test
Evaluates the model on ECP test.

For ECP scores (LAMR for every class): 

    python output_eval.py --weights_path misc/yolov3_ckpt_current_50.pth --conf_thres 0.90
    python from_ecpb/eval.py
    
Results will be stored in from_ecpb/results
    
For classic scores (mAP, F1, precision, recall)
     
TODO add pipeline to readme

## Train
```
$ train.py [-h] [--epochs EPOCHS][--batch_size BATCH_SIZE]
                [--gradient_accumulations N_ACCUMULATIONS]
                [--model_def MODEL_DEF][--data_config CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] 
                [--img_size IMG_SIZE][--n_cpu N_CPU]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP][--ECP BOOL] 
                [--multiscale_training MULTISCALE_TRAINING]
                [--optim OPTIM][eval_batch_lim LIM] 
                [--freeze_backbone_until EPOCHS][--lr LR]
                [--name SESSION][--start_epoch EPOCH]
                [--logger BOOL][--metric BOOL]
                [--data DATAPATH][--town SUBSET] 
               
```

#### Example
```
$ python train.py --epochs 200 --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --checkpoint_interval 5 --metric 1 --name scheduLR --freeze_backbone_until 2 --optim adam
```

#### Training log
```
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
Filtered Loss 4.034
Learning rate 0.002
---- ETA 0:35:48.821929
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
$ tensorboard --logdir='logs' --port=6006 --host 0.0.0.0 (if from cloud)
```

Tensorboard will instantiate a different log for every run thanks to the parameter --name.
At every checkpoint evaluation, the images present in  the folder data/samples will be also demoed and saved in tensorboard

#### Log

The log file can be downloaded via this link : 
https://drive.google.com/drive/folders/1Z9u35DOLv52WUHocVWuPC5kcMPzbaEXf?usp=sharing

## Credit

[PyTorch-YOLOv3: minimal implementation ](https://github.com/eriklindernoren/PyTorch-YOLOv3)


### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
