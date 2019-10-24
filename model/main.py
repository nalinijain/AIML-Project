import time

import torch
import torch.nn as nn

import numpy

from torch_model import build_model
from data import build_IMU_data
from defaults import cfg


def train_one_epoch(model, data, target):
    t1 = time.time()
    for _ in range(data.shape[0]):
        x = data[_]
        output = model(x)
        import pdb; pdb.set_trace()
        if (_+1)%20 == 0:
            print("%d iteration has passed :: %0.2f sec" %(_+1, time.time()-t1))
            t1 = time.time()

        


def do_train():
    is_train = True
    # Load Model
    gyro_model = build_model()
    epoch = cfg.NUM_EPOCH
    for i in range(epoch):
        IMU = build_IMU_data(is_train)
        data = IMU.data.to(cfg.MODEL_DEVICE)
        data = data.reshape(data.shape[0], 1, -1, data.shape[-1])
        target = IMU.label.to(cfg.MODEL_DEVICE)
        train_one_epoch(gyro_model, data, target)


if __name__ == '__main__':
    do_train()
    
