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
        target = target[:, :, :]
        loss = torch.mean(abs(target - output))

        return loss

def build_loss_function(cfg):
    return Loss_Evaluator(cfg)