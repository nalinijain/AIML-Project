import torch
import torch.nn as nn

class Loss_Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Loss_Evaluator, self).__init__()
        self.cfg = cfg

    def L2_loss(self, output, target):
        '''
        This loss function is to penalize the differences
        between predicted joints and ground truth
        '''
        conf = target[:, :, -1]
        target = target[:, :, :-1]/100
        diff = torch.sum((target-output)**2, dim=2)
        diff = torch.mul(diff, conf)
        loss = torch.sum(diff)

        return loss

def build_loss_function(cfg):
    return Loss_Evaluator(cfg)