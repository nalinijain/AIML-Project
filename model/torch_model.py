import torch
from torch import nn

from defaults import cfg
from torch_layer import BottleNeck, BottleNeck_stride

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.input_channel = len(cfg.TRAIN_PART)*6
        self.device = cfg.MODEL_DEVICE
        self.act_fn = nn.ReLU()
        self.ref_channel = cfg.REF_CHANNEL
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_channel, self.ref_channel, 3, 1, 1),
            self.act_fn, nn.MaxPool1d(3, 1, 1),
        )
        self.layer2 = nn.Sequential(
            BottleNeck(self.ref_channel, self.ref_channel, self.ref_channel*4, self.act_fn),
            BottleNeck(self.ref_channel*4, self.ref_channel, self.ref_channel*4, self.act_fn),
            BottleNeck(self.ref_channel*4, self.ref_channel, self.ref_channel*4, self.act_fn),
        )
        self.layer3 = nn.Sequential(
            BottleNeck(self.ref_channel*4, self.ref_channel*2, self.ref_channel*8, self.act_fn),
            BottleNeck(self.ref_channel*8, self.ref_channel*2, self.ref_channel*8, self.act_fn),
            BottleNeck(self.ref_channel*8, self.ref_channel*2, self.ref_channel*8, self.act_fn),
            BottleNeck_stride(self.ref_channel*8, self.ref_channel*2, self.ref_channel*8, self.act_fn),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(self.ref_channel*8, self.ref_channel*8, 3, 1, 1),
            nn.BatchNorm1d(self.ref_channel*8), self.act_fn,
            nn.Conv1d(self.ref_channel*8, self.ref_channel*4, 3, 1, 1),
            nn.BatchNorm1d(self.ref_channel*4), self.act_fn,
            nn.ConvTranspose1d(self.ref_channel*4, 6*len(cfg.TRAIN_LABEL_PART), 3, 2)
        )

        '''
        self.layer4 = nn.Sequential(
            BottleNeck(self.ref_channel*8, self.ref_channel*4, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*4, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*4, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*4, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*4, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*4, self.ref_channel*16, self.act_fn, stride=2),
        )
        '''
        self.to(self.device)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(3, -1, x.shape[-1])
        return x



class gyroCNN(nn.Module):
    def __init__(self):
        super(gyroCNN, self).__init__()
        self.device = cfg.MODEL_DEVICE
        self.input_channel = len(cfg.TRAIN_PART)
        self.output_channel = 15
        self.layer1 = nn.Sequential(                        # Input size = 3 * 6 * 256
                nn.Conv1d(self.input_channel, 32, 3, 1, 1), # Output size = 3 * 32 * 256
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, 2))                         # Output size = 3 * 32 * 128
        self.layer2 = nn.Sequential(
                nn.Conv1d(32, 256, 3, 1, 1),                # Output size = 3 * 256 * 128
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2,2))                          # Output size = 3 * 64 * 64
        self.layer3 = nn.Sequential(
                nn.ConvTranspose1d(256, 64, 3, 2),       # Output size = 3 * 64 * 128
                nn.BatchNorm1d(64),
                nn.ReLU())
        self.layer4 = nn.Sequential(
                nn.ConvTranspose1d(64, self.output_channel, 3, 2, 1),        # Output size = 3 * 15 * 256
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU())
        self.to(self.device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x




def build_model():
    if cfg.ARCHITECTURE == "CNN":
        return gyroCNN()
    if cfg.ARCHITECTURE == "ResNet":
        return ResNet()
    else:
        raise AssertionError("No such architecture exists")
