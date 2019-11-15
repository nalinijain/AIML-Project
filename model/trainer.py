import time, sys, pickle, os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from NeuralNet import jointCNN
from data import build_IMU_data, build_keypoint_data

#sys.path.append('/home/soyongs/research/codes/AIML-Project/')
from defaults import cfg


class Trainer():
    def __init__(self):
        self.num_epoch = cfg.NUM_EPOCH
        self.lr = cfg.LEARNING_RATE/cfg.BATCH_SIZE
        self.model = jointCNN(cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.epoch = 1
        self.iteration = 1
        self.output_dir = cfg.OUTPUT_DIR
        self.checkpoint = cfg.CHECKPOINT
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR, comment=cfg.ARCHITECTURE)
        
    def do_train(self):
        if cfg.USE_PKL:
            with open(cfg.PKL_DIR + 'accel.pkl', 'rb') as accel_file:
                accel = pickle.load(accel_file)    
            with open(cfg.PKL_DIR + 'gyro.pkl', 'rb') as gyro_file:
                gyro = pickle.load(gyro_file)
            with open(cfg.PKL_DIR + 'label.pkl', 'rb') as label_file:
                label = pickle.load(label_file)        
            with open(cfg.PKL_DIR + 'kp.pkl', 'rb') as kp_file:
                kp = pickle.load(kp_file)

        else:
            IMU = build_IMU_data(cfg)
            accel = IMU.accel
            gyro = IMU.gyro
            keypoint = build_keypoint_data(cfg)
            kp = keypoint.input
            label = keypoint.ground_truth

            for i in range(len(accel)):
                accel[i] = torch.transpose(accel[i].reshape(-1, cfg.SEQUENCE_LENGTH, 1, 3), 1, 3)
                gyro[i] = torch.transpose(gyro[i].reshape(-1, cfg.SEQUENCE_LENGTH, 1, 3), 1, 3)
                label[i] = label[i].reshape(-1, int(cfg.SEQUENCE_LENGTH/5), 1, 4)

        for _ in range(self.num_epoch):
            self.train_one_epoch([accel, gyro, kp], label)
            self.epoch += 1

    def train_one_epoch(self, inputs, label):
        accel, gyro, kp = inputs
        
        for i in range(len(accel)):
            for j in range(0, accel[i].shape[0]-cfg.BATCH_SIZE, cfg.BATCH_SIZE):
                loss = 0
                print_loss = {}
                for b in range(cfg.BATCH_SIZE):
                    losses = self.model([accel[i][j+b], gyro[i][j+b]], kp[i][j+b], label[i][j+b])
                    for l in losses:
                        loss += losses[l]
                        try:
                            print_loss[l] += losses[l]
                        except:
                            print_loss[l] = losses[l]
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print_loss['total_loss'] = loss
                self.print_and_log(print_loss)
                self.iteration += 1


    def save_model(self):
        check_point = ({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': cfg.ARCHITECTURE,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer
        })
        
        file_name = '%08d.pt'%self.iteration
        torch.save(check_point, os.path.join(self.output_dir, file_name))


    def print_and_log(self, losses):
        loss_dict = []
        for l in losses:
            loss_dict.append(l)
            loss_ = int(losses[l].item()*10**4)/10**4
            loss_dict.append(loss_)
        if(self.iteration % 100) == 0:
            print("Training progress : Total Iteration %d | Epoch [%d|%d] | "%(self.iteration, self.epoch, self.num_epoch)
            , " ".join(str(l) for l in loss_dict))
        if self.iteration%self.checkpoint == 0:
            print("Checkpoint saving | iteration = %d" %self.iteration)
            self.save_model()

        for l in losses:
            self.writer.add_scalar(l, losses[l].item(), self.iteration)
        


if __name__ == '__main__':
    trainer = Trainer()
    trainer.do_train()