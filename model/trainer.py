import time, sys, pickle, os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from NeuralNet import jointCNN
from data import build_IMU_data

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
        for _ in range(self.num_epoch):
            
            if cfg.USE_PKL:
                open_input = open('../Dataset/input.pkl', 'rb')
                open_label = open('../Dataset/label.pkl', 'rb')
                input_data = pickle.load(open_input)
                label = pickle.load(open_label)
                open_input.close()
                open_label.close()
            else:
                IMU = build_IMU_data(cfg)
                data = IMU.data
                input_data = data[:,:,0,:].reshape(-1, 6, 1, cfg.SEQUENCE_LENGTH)
                label = data[:,:,1:,:]

            self.train_one_epoch(input_data, label)
            self.epoch += 1

    def train_one_epoch(self, inputs, label):
        gyro = inputs[:, :3, :, :]
        accel = inputs[:, 3:, :, :]
        rand_perm = torch.randperm(gyro.shape[0])
        accel = torch.tensor(accel[rand_perm]).cuda().float()
        gyro = torch.tensor(gyro[rand_perm]).cuda().float()
        label = torch.tensor(label[rand_perm]).cuda().float()

        for j in range(0, accel.shape[0]-cfg.BATCH_SIZE, cfg.BATCH_SIZE):
            loss = 0
            print_loss = {}
            for b in range(cfg.BATCH_SIZE):
                losses = self.model([accel[j+b], gyro[j+b]], label[j+b])
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