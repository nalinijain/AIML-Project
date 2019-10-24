import torch
import torch.nn as nn


def conv_1_1_y(input_channel, output_channel, act_fn):
    model = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1),
        nn.BatchNorm1d(output_channel), act_fn,
    )
    return model

def conv_1_2_y(input_channel, output_channel, act_fn):
    model = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=2),
        nn.BatchNorm1d(output_channel), act_fn,
    )
    return model

def conv_3_1_y(input_channel, output_channel, act_fn):
    model = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(output_channel), act_fn,
    )
    return model

def conv_1_1_n(input_channel, output_channel):
    model = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1),
        nn.BatchNorm1d(output_channel), 
    )
    return model

def conv_1_2_n(input_channel, output_channel):
    model = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=2),
        nn.BatchNorm1d(output_channel), 
    )
    return model


class BottleNeck(nn.Module):
    
    def __init__(self, input_channel, hidden_channel, output_channel, act_fn):
        super(BottleNeck,self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.act_fn = act_fn
        self.layer = nn.Sequential(
            conv_1_1_y(input_channel, hidden_channel, act_fn),
            conv_3_1_y(hidden_channel, hidden_channel, act_fn),
            conv_1_1_n(hidden_channel, output_channel),
        )
        self.downsample = nn.Conv1d(input_channel, output_channel, 1, 1)
    
    def forward(self, x):
        if self.input_channel == self.output_channel:
            shortcut = x
        else:
            shortcut = self.downsample(x)
        x = self.layer(x)
        x = x + shortcut
        x = self.act_fn(x)
        
        return x

class BottleNeck_stride(nn.Module):
    
    def __init__(self, input_channel, hidden_channel, output_channel, act_fn):
        super(BottleNeck_stride,self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.act_fn = act_fn
        self.layer = nn.Sequential(
            conv_1_2_y(input_channel, hidden_channel, act_fn),
            conv_3_1_y(hidden_channel, hidden_channel, act_fn),
            conv_1_1_n(hidden_channel, output_channel),
        )
        self.downsample = nn.Conv1d(input_channel, output_channel, 1, 2)
    
    def forward(self, x):
        shortcut = self.downsample(x)
        x = self.layer(x)
        x = x + shortcut
        x = self.act_fn(x)
        
        return x

