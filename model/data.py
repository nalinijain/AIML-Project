import pandas as pd
import numpy as np
import torch
import random, pickle

import os, sys
sys.path.append('/home/soyongs/research/codes/joint-CNN/')
from utils.data_utils import *

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
                    #num_exp = int(len(files)/2)
                    num_exp = 14        # Maximum experiment number
                    
                    actual_num_exp = -1

                    for i in range(num_exp):
                        check_unsynced = d + '_' +str(i+1) + '_' + str(id+1)
                        if self.cfg.UNSYNCED_LIST.count(check_unsynced):
                            continue

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

        return data        
        
        #Drop from the back to the size of data is multiple of sequence length
        '''
        rem = data.shape[0] % self.cfg.SEQUENCE_LENGTH
        if rem != 0:
            drop_remain_index = data.index[:(-1)*rem]
            return data.reindex(drop_remain_index)
        

        else:
            return data        
        '''
        

class keypoint():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DATA_DEVICE
        self.training_data = []
        #self.input, self.ground_truth = self.load_data()
        self.ground_truth = self.load_data()

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
                #exp_folder += self.cfg.KEYPOINTS_DEFAULT
                #joints, _ = kp_loader(exp_folder)
                
                joints, _ = refined_kp_loader(exp_folder)
                
                if len(joints) == 0:
                    continue
                
                joints = self.zero_conf(joints)
                for id in range(2):
                    joints[id] = joints[id][:,self.cfg.KEYPOINT_PART,:]
                    #joints[id] = self.drop_out(joints[id])
                    joints[id].reshape(-1, 4)
                joints[0] = torch.tensor(joints[0])
                joints[1] = torch.tensor(joints[1])
                if self.device == 'cuda:0':
                    joints[0] = joints[0].to(self.device).float()
                    joints[1] = joints[1].to(self.device).float()
                
                check_unsynced_0 = d + '_' + str(i+1) + '_1'
                check_unsynced_1 = d + '_' + str(i+1) + '_2'
                subject_message = ''
                if self.cfg.UNSYNCED_LIST.count(check_unsynced_0) == 0:
                    subject_1.append(joints[0])
                    subject_message += '1'
                if self.cfg.UNSYNCED_LIST.count(check_unsynced_1) == 0:
                    subject_2.append(joints[1])
                    if subject_message == '1':
                        subject_message += ', 2'
                    else:
                        subject_message = '2'
                
                print("Completed loading keypoint data | date : %s, exp : %d, subject : %s" 
                %(d, i+1, subject_message))
        
        self.training_data = subject_1 + subject_2
        #input_data, output_data = self.separate_input_output(self.training_data)
        output_data = self.separate_input_output(self.training_data)

        return output_data
        #return input_data, output_data

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
        
        output_data = []
        interval = int(self.cfg.SEQUENCE_LENGTH/5)
        for i in range(len(data)):
            tmp_output = data[i]
            if tmp_output.shape[0]%interval != 0:
                tmp_output = tmp_output[:-(tmp_output.shape[0]%interval)]
            
            output_data.append(tmp_output)

        return output_data

    def drop_out(self, data):
        """
        This function is to drop back data.
        """

        rem = data.shape[0] % int(self.cfg.SEQUENCE_LENGTH/5)
        if rem != 0:
            return data[:(-1)*rem]
        else:
            return data

def data_augmentation(cfg, init_pose, label):
    rand = torch.normal(0, 100, size=(init_pose.shape[0], 1, 3)).cuda().float()
    init_pose[:, :, :-1] = init_pose[:, :, :-1] + rand
    label[:, :, :, :-1] = label[:, :, :, :-1] + rand.reshape(-1, 1, 1, 3)

    return init_pose, label

def make_mini_batch(cfg, data):
    # Size synthesizing
    for i in range(len(data)):
        data[i] = torch.cat(data[i], dim=0)
    
    if cfg.DATA_AUGMENTATION:
        data[2], data[3] = data_augmentation(cfg, data[2], data[3])

    assert (data[0].shape[0] == data[1].shape[0] and
        data[1].shape[0] == data[2].shape[0] and 
        data[2].shape[0] == data[3].shape[0])

    # Random permuting
    rand = torch.randperm(data[0].shape[0])
    for i in range(len(data)):
        data[i] = data[i][rand]
    
    # Batch generating
    rem = data[0].shape[0] % cfg.BATCH_SIZE
    if rem != 0:
        for i in range(len(data)):
            data[i] = data[i][:(-1)*rem]
            try:
                data[i] = data[i].reshape(-1, cfg.BATCH_SIZE, data[i].shape[1], data[i].shape[2], data[i].shape[3])
            except:
                data[i] = data[i].reshape(-1, cfg.BATCH_SIZE, data[i].shape[1], data[i].shape[2])

    return data


def generate_init_pose(cfg, label):
    init_pose_idx = []
    interval = int(cfg.SEQUENCE_LENGTH/5)
    for i in range(cfg.NUM_INIT_POSES):
        tmp_idx = [j * interval + i for j in range(int(label.shape[0]/interval)) if j*interval < label.shape[0]]
        init_pose_idx += tmp_idx
    
    init_pose_idx.sort()
    init_pose = label[init_pose_idx].reshape(-1, cfg.NUM_INIT_POSES, len(cfg.KEYPOINT_PART), 4)
    
    return init_pose


def data_preprocess(cfg, accel, gyro, label):
    print("Data for new epoch loaded. Preprocessing .... ")
    
    init_pose = []
    for i in range(len(accel)):
        kp_drop = int(random.random()*10000 % int(cfg.SEQUENCE_LENGTH/5))
        label[i] = label[i][kp_drop:]

        imu_drop = 5 * kp_drop + 5 * cfg.NUM_INIT_POSES
        accel[i] = accel[i][imu_drop:]
        gyro[i] = gyro[i][imu_drop:]

        imu_rem = accel[i].shape[0] % cfg.SEQUENCE_LENGTH
        if imu_rem != 0:
            accel[i] = accel[i][:(-1)*imu_rem]
            gyro[i] = gyro[i][:(-1)*imu_rem]
        
        init_pose.append(generate_init_pose(cfg, label[i]))
        label[i] = label[i][cfg.NUM_INIT_POSES:]

        kp_rem = label[i].shape[0] % int(cfg.SEQUENCE_LENGTH/5)
        if kp_rem != 0:
            label[i] = label[i][:(-1)*kp_rem]

        accel[i] = torch.transpose(accel[i].reshape(-1, cfg.SEQUENCE_LENGTH, 3*len(cfg.TRAIN_PART)), 1, 2)
        gyro[i] = torch.transpose(gyro[i].reshape(-1, cfg.SEQUENCE_LENGTH, 3*len(cfg.TRAIN_PART)), 1, 2)
        label[i] = label[i].reshape(-1, int(cfg.SEQUENCE_LENGTH/5), len(cfg.KEYPOINT_PART), 4)

        init_pose[i] = init_pose[i][:accel[i].shape[0]]
        label[i] = label[i][:accel[i].shape[0]]
    
    [accel, gyro, init_pose, label] = make_mini_batch(cfg, [accel, gyro, init_pose, label])
    
    return accel, gyro, init_pose, label

def to_cpu(data):
    for d in data:
        for i in range(len(data[0])):
            d[i] = d[i].cpu()

    return data

def load_data(cfg):
    if cfg.USE_PKL:
        with open(cfg.PKL_DIR + 'accel.pkl', 'rb') as accel_file:
            accel = pickle.load(accel_file)
        with open(cfg.PKL_DIR + 'gyro.pkl', 'rb') as gyro_file:
            gyro = pickle.load(gyro_file)
        with open(cfg.PKL_DIR + 'label.pkl', 'rb') as label_file:
            label = pickle.load(label_file)
                
    else:
        keypoint = build_keypoint_data(cfg)
        label = keypoint.ground_truth
        IMU = build_IMU_data(cfg)
        accel = IMU.accel
        gyro = IMU.gyro
        
        accel_open = open("accel.pkl", "wb")
        gyro_open = open("gyro.pkl", "wb")
        label_open = open("label.pkl", "wb")
        pickle.dump(accel, accel_open)
        pickle.dump(gyro, gyro_open)
        pickle.dump(label, label_open)
        import pdb; pdb.set_trace()

        

    accel, gyro, init_pose, label = data_preprocess(cfg, accel, gyro, label)
    return accel, gyro, init_pose, label

def prepare_evalset(cfg):
    pass


def build_IMU_data(cfg):
    return IMU(cfg)

def build_keypoint_data(cfg):
    return keypoint(cfg)