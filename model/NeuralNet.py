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
        self.model = build_model(cfg)
        self.loss = build_loss_function(cfg)
        self.to(self.device)

    def forward(self, imu_data, kp_data, target):
        losses = {}
        imu, kp = self.prep(imu_data, kp_data[:,:-1])
        input = torch.cat((kp, imu), dim=2)
        output = self.model(input)
        l2_loss = self.loss.L2_loss(output, target)
        losses['l2_loss'] = l2_loss
        
        return losses
        