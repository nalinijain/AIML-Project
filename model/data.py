import pandas as pd
import numpy as np
import torch
import random

import os, sys
from data_utils import *

class IMU():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DATA_DEVICE
        self.is_random = cfg.RANDOM_SAMPLING
        self.gyro, self.accel = self.load_data()
        
        # TODO: Build test dataset loader that might be used in future research

    def load_data(self):
        
        print("Start loading IMU data.")
        gyro_data = []
        accel_data = []
        _, subs, _ = next(os.walk(self.cfg.IMU_DIR))
        
        for id in range(2):
            subject_folder = os.path.join(self.cfg.IMU_DIR, subs[id])
            
            for d in self.cfg.DATA_DATE:    
                print("============ IMU data loading | date : %s, subject : %d ============" %(d, id+1))
                gyro_exp_data = []
                accel_exp_data = []
                
                for part in self.cfg.TRAIN_PART:
                    date_folder = os.path.join(subject_folder, d)
                    part_folder = os.path.join(date_folder, part)
                    _, _, files = next(os.walk(part_folder))
                    
                    assert len(files)%2 == 0
                    num_exp = 14        # Maximum experiment number
                    
                    actual_num_exp = -1

                    for i in range(num_exp):
                        accel_name = 'accel_exp' + str(i+1) + '.csv'
                        gyro_name = 'gyro_exp' + str(i+1) + '.csv'
                        try:
                            accel = pd.read_csv(
                                os.path.join(part_folder, accel_name), 
                                index_col='Timestamp (microseconds)')
                            gyro = pd.read_csv(
                                os.path.join(part_folder, gyro_name), 
                                index_col='Timestamp (microseconds)')
                            
                            actual_num_exp += 1
                        
                        except:
                            continue

                        gyro = self.drop_out(gyro)
                        gyro = self.rename_columns(gyro, part)
                        accel = self.drop_out(accel)
                        accel = self.rename_columns(accel, part)
                        
                        try:
                            gyro_exp_data[actual_num_exp] = gyro_exp_data[actual_num_exp].join(gyro, how='outer')
                        except:
                            gyro_exp_data.append(gyro)
                        try:
                            accel_exp_data[actual_num_exp] = accel_exp_data[actual_num_exp].join(accel, how='outer')
                        except:
                            accel_exp_data.append(accel)
                
                    if part == self.cfg.TRAIN_PART[0]:
                        print("%d experiments loaded" %(actual_num_exp+1))

                gyro_data += self.reshape_data(gyro_exp_data)
                accel_data += self.reshape_data(accel_exp_data)
        
        return gyro_data, accel_data

    def reshape_data(self, data_sequence):
        """
        This functions is to reshape and convert pandas DataFrame to torch tensor
        Data shape will be (-1, N, C, L) while: 
        -1 varies with the size of IMU sensor data per experiment,
        N is 3 X num_of_parts,
        C is num_of_channel which is 1,
        L is length_of_sequence which is defined in configuration file.
        """
        
        for i in range(len(data_sequence)):
            rs = np.array(data_sequence[i])
            
            '''
            Random dropout needs reshape in the future.

            rs = rs.reshape(-1, self.cfg.SEQUENCE_LENGTH, rs.shape[1])
            rs = np.transpose(rs, (0, 2, 1))
            rs = rs.reshape(rs.shape[0], rs.shape[1], 1, -1)
            '''

            rs = torch.tensor(rs)
            if self.device == 'cuda:0':
                rs = rs.to(self.device).float()
            data_sequence[i] = rs
                

        return data_sequence

    def rename_columns(self, data, part):
        """
        This function is to change the column names of data.
        This avoids name overlap and enalbes data to be joined.
        """

        new_columns = []
        new_columns.append('X_' + part)
        new_columns.append('Y_' + part)
        new_columns.append('Z_' + part)

        new_columns = pd.Index(new_columns)
        data.columns = new_columns

        return data

    def drop_out(self, data):
        """
        This function is to drop front and back data.
        """        
        
        #Drop from the back to discard invalid data
        dropped_index = data.index[:(-1)*self.cfg.DATA_DROP]
        data = data.reindex(dropped_index)
        
        #Drop from the front to generate different sequence combination
        if self.is_random:
            rand = int(random.random() * 10000) % (self.cfg.SEQUENCE_LENGTH/5) * 5
            random_drop_index = data.index[rand:]
            data = data.reindex(random_drop_index)

        #Drop from the back to the size of data is multiple of sequence length
        rem = data.shape[0] % self.cfg.SEQUENCE_LENGTH
        if rem != 0:
            drop_remain_index = data.index[:(-1)*rem]
            return data.reindex(drop_remain_index)

        else:
            return data        
        

class keypoint():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DATA_DEVICE
        self.training_data = []
        self.input, self.ground_truth = self.load_data()

    def load_data(self):
        print("Start loading keypoint data.")
        subject_1, subject_2 = [[], []]

        for d in self.cfg.DATA_DATE:
            date_folder = os.path.join(self.cfg.KEYPOINTS_DIR, d)
            _, exps, _ = next(os.walk(date_folder))
            exps.sort()
            num_exp = len(exps)
            for i in range(num_exp):
                exp_folder = os.path.join(date_folder, exps[i])
                
                print("============ Keypoint data loading | date : %s, exp : %d ============" %(d, i+1))
                joints, _ = kp_loader(exp_folder)
                
                if len(joints) == 0:
                    continue
                
                joints = self.zero_conf(joints)
                for id in range(2):
                    joints[id] = joints[id][:,self.cfg.KEYPOINT_PART,:]
                    joints[id] = self.drop_out(joints[id])
                    #joints[id].reshape(-1, int(self.cfg.SEQUENCE_LENGTH/5), 4)
                    joints[id].reshape(-1, 4)
                joints[0] = torch.tensor(joints[0])
                joints[1] = torch.tensor(joints[1])
                if self.device == 'cuda:0':
                    joints[0] = joints[0].to(self.device).float()
                    joints[1] = joints[1].to(self.device).float()
                
                check_unsynced_0 = d + '_' + str(i+1) + '_1'
                check_unsynced_1 = d + '_' + str(i+1) + '_2'
                if self.cfg.UNSYNCED_LIST.count(check_unsynced_0) == 0:
                    subject_1.append(joints[0])
                if self.cfg.UNSYNCED_LIST.count(check_unsynced_1) == 0:
                    subject_2.append(joints[1])
        
        self.training_data = subject_1 + subject_2
        input_data, output_data = self.separate_input_output(self.training_data)
        return input_data, output_data

    def zero_conf(self, joints):
        for i in range(len(joints)):
            conf = joints[i][:, :, -1]
            conf = np.where(conf == -1, 0, conf)
            joints[i][:, :, -1] = conf
        
        return joints

    def separate_input_output(self, data):
        """
        This function is to classify keypoint data 
        that goes as input and ground truth to compare with output.
        """
        
        input_data, output_data = [[], []]
        interval = int(self.cfg.SEQUENCE_LENGTH/5)
        for i in range(len(data)):
            input_idx = [j*interval for j in range(int(data[i].shape[0]/interval)) if j*interval < data[i].shape[0]]
            #input_data.append(data[i][input_idx][:, :, :-1])   confidence value is necessary
            input_data.append(data[i][input_idx])

            tmp_output = data[i][1:]
            if tmp_output.shape[0]%interval != 0:
                tmp_output = tmp_output[:-(tmp_output.shape[0]%interval)]
            
            #tmp_output = tmp_output.reshape(-1, interval, tmp_output.shape[1], tmp_output.shape[2])
            output_data.append(tmp_output)

        return input_data, output_data

    def drop_out(self, data):
        """
        This function is to drop back data.
        """

        rem = data.shape[0] % int(self.cfg.SEQUENCE_LENGTH/5)
        if rem != 0:
            return data[:(-1)*rem]
        else:
            return data



def build_IMU_data(cfg):
    return IMU(cfg)

def build_keypoint_data(cfg):
    return keypoint(cfg)