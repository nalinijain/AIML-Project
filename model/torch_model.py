import torch
from torch import nn

from defaults import cfg
from torch_layer import BottleNeck, BottleNeck_stride


class LSTM(nn.Module):
    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.device = cfg.MODEL_DEVICE
        self.num_layer = 3
        self.input_size = 6
        self.hidden_size = 32
        self.layer = nn.LSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layer, bidirectional=True)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size*cfg.SEQUENCE_LENGTH*2, 128),
            nn.BatchNorm1d(1), nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(1), nn.ReLU(),
            nn.Linear(256, cfg.SEQUENCE_LENGTH*2*3),
        )
        self.to(self.device)

    def forward(self, x):
        x = x.transpose(2, 0)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layer*2, 1, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layer*2, 1, self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.layer(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out.reshape(1, 1, -1)
        out = self.fc_layer(out)
        out = out.reshape(-1, 2, cfg.SEQUENCE_LENGTH)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        
        self.device = cfg.MODEL_DEVICE
        
        self.act_fn = nn.ReLU()
        self.ref_channel = cfg.REF_CHANNEL

        self.input_channel = len(cfg.TRAIN_PART) * 6
        self.output_channel = self.ref_channel * 2
        self.linear_input = self.output_channel * int((cfg.SEQUENCE_LENGTH-11)/2)
        self.linear_output = len(cfg.KEYPOINT_PART) * int(cfg.SEQUENCE_LENGTH/5) * 3
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_channel, self.ref_channel, 3, 1),
            self.act_fn, nn.MaxPool1d(3, 1),
        )
        self.layer2 = nn.Sequential(
            BottleNeck(self.ref_channel, self.ref_channel, self.ref_channel*4, self.act_fn),
            BottleNeck(self.ref_channel*4, self.ref_channel, self.ref_channel*4, self.act_fn),
            BottleNeck(self.ref_channel*4, self.ref_channel, self.ref_channel*4, self.act_fn),
        )
        self.layer3 = nn.Sequential(
            BottleNeck(self.ref_channel*4, self.ref_channel*2, self.ref_channel*8, self.act_fn),
            BottleNeck(self.ref_channel*8, self.ref_channel*2, self.ref_channel*16, self.act_fn),
            BottleNeck(self.ref_channel*16, self.ref_channel*2, self.ref_channel*32, self.act_fn),
            BottleNeck_stride(self.ref_channel*32, self.ref_channel*2, self.ref_channel*32, self.act_fn),
        )
        self.layer4 = nn.Sequential(
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
            BottleNeck(self.ref_channel*32, self.ref_channel*16, self.ref_channel*32, self.act_fn),
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(self.ref_channel*32, self.ref_channel*32, 3, 1),
            nn.BatchNorm1d(self.ref_channel*32), self.act_fn,
            nn.Conv1d(self.ref_channel*32, self.ref_channel*16, 3, 1),
            nn.BatchNorm1d(self.ref_channel*16), self.act_fn,
            nn.Conv1d(self.ref_channel*16, self.ref_channel*8, 3, 1, 1),
            nn.BatchNorm1d(self.ref_channel*8), self.act_fn,
            nn.Conv1d(self.ref_channel*8, self.ref_channel*2, 3, 1, 1),
            nn.BatchNorm1d(self.ref_channel*2), self.act_fn,
        )

        self.fcn = nn.Sequential(
            nn.Linear(self.linear_input, 128),
            self.act_fn,
            nn.Linear(128, 64),
            self.act_fn,
            nn.Linear(64, self.linear_output),
        )

        self.to(self.device)
    
    def forward(self, x):
        if self.training:
            batch_size = self.cfg.BATCH_SIZE
        else:
            batch_size = 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(batch_size, -1)
        x = self.fcn(x)
        x = x.reshape(
            batch_size,
            int(self.cfg.SEQUENCE_LENGTH/int(self.cfg.SAMPLING_FREQ/25)), 
            len(self.cfg.KEYPOINT_PART), -1)
        return x



class Preprocess(nn.Module):
    def __init__(self, cfg):
        super(Preprocess, self).__init__()
        self.cfg = cfg
        self.device = cfg.MODEL_DEVICE
        
        self.imu_input = len(cfg.TRAIN_PART) * 3
        self.imu_hidden = 16
        self.imu_output = len(cfg.TRAIN_PART) * 3

        self.kp_inpu = len(cfg.KEYPOINT_PART)*3
        self.kp_output = len(cfg.TRAIN_PART)*6
        
        self.input_feature = len(cfg.KEYPOINT_PART)*3*cfg.NUM_INIT_POSES
        self.output_feature = len(cfg.TRAIN_PART)*6

        self.accel_prep = nn.Sequential(
            nn.Conv1d(self.imu_input, self.imu_hidden, 3, 1, 1),
            nn.BatchNorm1d(self.imu_hidden), nn.ReLU(),
            nn.Conv1d(self.imu_hidden, self.imu_output, 3, 1, 1),
            nn.BatchNorm1d(self.imu_output), nn.ReLU(),
        )

        self.gyro_prep = nn.Sequential(
            nn.Conv1d(self.imu_input, self.imu_hidden, 3, 1, 1),
            nn.BatchNorm1d(self.imu_hidden), nn.ReLU(),
            nn.Conv1d(self.imu_hidden, self.imu_output, 3, 1, 1),
            nn.BatchNorm1d(self.imu_output), nn.ReLU(),
        )
        self.kp_prep = nn.Sequential(
            nn.Linear(self.input_feature, 128),
            nn.BatchNorm1d(1), nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(1), nn.ReLU(),
            nn.Linear(64, self.output_feature)
        )
        self.to(self.device)

    def forward(self, imu, kp):
        if self.training:
            batch_size = self.cfg.BATCH_SIZE
        else:
            batch_size = 1

        accel, gyro = imu
        accel = self.accel_prep(accel)
        gyro = self.gyro_prep(gyro)
        imu = torch.cat((accel, gyro), dim=1)
        
        kp = kp.reshape(batch_size, 1, -1)
        kp = self.kp_prep(kp)
        kp = kp.reshape(batch_size, -1, 1)

        return imu, kp



def build_preprocess(cfg):
    return Preprocess(cfg)

def build_model(cfg):
    if cfg.ARCHITECTURE == "ResNet":
        return ResNet(cfg)
    elif cfg.ARCHITECTURE == "LSTM":
        return LSTM(cfg)
    else:
        raise AssertionError("No such architecture exists")