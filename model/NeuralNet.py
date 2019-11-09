from torch_model import build_model, build_preprocess

import torch
from torch import nn
from loss_evaluator import build_loss_function
import numpy as np

class jointCNN(nn.Module):
    '''
    Joint-CNN
    '''

    def __init__(self, cfg):
        super(jointCNN, self).__init__()
        
        self.device = cfg.MODEL_DEVICE
        self.prep = build_preprocess(cfg)
        self.gyro_model = build_model(cfg)
        self.accel_model = build_model(cfg)
        self.loss = build_loss_function(cfg)
        self.to(self.device)

    def forward(self, imu_data, target):
        losses = {}
        imu = self.prep(imu_data)
        accel_output = self.accel_model(imu)
        gyro_output = self.gyro_model(imu)
        accel_target = target[3:, :, :]
        gyro_target = target[:3, :, :]
        l2_loss_accel = self.loss.L2_loss(accel_output, accel_target)
        l2_loss_gyro = self.loss.L2_loss(gyro_output, gyro_target)
        losses['accel_loss'] = l2_loss_accel
        losses['gyro_loss'] = l2_loss_gyro
        
        return losses
        