import torch
import torch.nn as nn

class Loss_Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Loss_Evaluator, self).__init__()
        self.cfg = cfg

    def l2_loss(self, output, target, init_conf=None):
        '''
        This loss function is to penalize the differences
        between predicted joints and ground truth
        '''

        losses = {}

        conf = target[:, :, :, -1]
        if self.cfg.CONSIDER_INIT_CONF:
            conf = conf * init_conf.reshape(self.cfg.BATCH_SIZE, 1, -1)
        target = target[:, :, :, :-1]/100                 # Change the joint location unit from cm to m


        diff = torch.sum((target - output)**2, dim=-1)
        diff = torch.mul(diff, conf)
        loss = diff.sum()

        loss = loss * self.cfg.L2_LW

        losses['l2_loss'] = loss

        return losses


def build_loss_function(cfg):
    return Loss_Evaluator(cfg)