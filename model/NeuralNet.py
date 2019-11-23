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

    def forward(self, imu_data, init_pose, target):
        losses = {}
        
        if init_pose.shape[-1] == 4:
            try:
                init_conf = init_pose[:,:,:,-1]
            except:
                import pdb; pdb.set_trace()
            init_conf = init_conf.prod(dim=1)
            init_pose = init_pose[:,:,:,:-1]
        

        imu, kp = self.prep(imu_data, init_pose)
        input = torch.cat((kp, imu), dim=2)
        output = self.model(input)
        l2_loss = self.loss.l2_loss(output, target, init_conf=init_conf)
        losses['l2_loss'] = l2_loss
        
        return losses
        