"""
This code is pipeline of loading and processing IMU data
Developer : Soyong Shin
"""

import os, time

from defaults import cfg

import numpy as np
import pandas as pd
import torch
import random

class IMUData():
    def __init__(self, is_train):
        self.is_train = is_train
        self.is_input = True
        self.device = cfg.DATA_DEVICE

        self.data_dir = cfg.IMU_DIR
        self.date = cfg.DATA_DATE
        self.subject = cfg.IMU_SET
        self.random_drop = int(random.random()*1000)%cfg.SEQUENCE_LENGTH
        
        if self.is_train:
            self.input_part = cfg.TRAIN_PART
            self.label_part = cfg.TRAIN_LABEL_PART
        else:
            self.input_part = cfg.TEST_PART
            self.label_part = cfg.TEST_PART


        self.data = self.load_data(True)
        self.label = self.load_data(False)
        if cfg.TENSOR_TYPE == "torchFloatTensor":
            self.data = torch.tensor(self.data).to(self.device)
            self.data = self.data.float()
            self.label = torch.tensor(self.label).to(self.device)
            self.label = self.label.float()
        
    
    def load_data(self, is_input):
        """
        This Code is basic data loading code for ML training.
        It works two times for every epochs in order to generate both input data and labels(targets / GT)
        """

        if is_input:
            print("Start loading input data ... /\/\/\//\/\/\//\/\/\/")
        else:
            print("Start loading labels ... /\/\/\//\/\/\//\/\/\/")
        
        if is_input:
            self.is_input = True
            num_features = 6
            part = self.input_part
        else:
            self.is_input = False
            num_features = 3
            part = self.label_part

        data = dict()

        for s in self.subject:
            
            print("Loading annotation file for Set%d" %s)
            
            _, sets, _ = next(os.walk(self.data_dir))
            subject_dir = os.path.join(self.data_dir, sets[s-1])
            anno_start, anno_end = self.load_annotation(subject_dir)
            for d in self.date:
                date_dir = os.path.join(subject_dir, d)
                for p in part:
                    
                    print("Loading data | date : %s (YYMMDD), part : %s" %(d, p))
                    accum_length = []
                    part_dir = os.path.join(date_dir, p)
                    gyro = pd.read_csv(part_dir+'/gyro.csv', index_col = 'Timestamp (microseconds)')
                    if self.is_input:
                        accel = pd.read_csv(part_dir+'/accel.csv', index_col = 'Timestamp (microseconds)')
                    else:
                        accel = []
                    time_start, time_end = self.segment_annotation(gyro, anno_start, anno_end)
                    for t in range(len(time_start)):
                        if self.is_input:
                            seg_data = self.segment_data(accel, gyro, time_start[t], time_end[t])
                        else:
                            seg_data = self.segment_data(accel, gyro, time_start[t], time_end[t])
                        re_data = self.resample_data(seg_data)
                        accum_length.append(re_data.shape[0])
                        try:
                            data[p] = data[p].append(re_data)
                        except:
                            data[p] = re_data
                    accum_length = np.array(accum_length).reshape(1,-1)
                    try:
                        length = np.vstack([length, accum_length])
                    except:
                        length = accum_length
                    
                    new_index = pd.RangeIndex(start = 1, stop = data[p].shape[0] + 1, step = 1)
                    data[p].index = new_index
                
                print("Size syncing... | date : %s (YYMMDD)" %d)

                data = self.size_syncing_data(length, data)

        for p in part:
            try:
                tensor = np.vstack([tensor, np.array(data[p].T).reshape(1,num_features,-1)])
            except:
                tensor = np.array(data[p].T).reshape(1,num_features,-1)
        tensor = (tensor.T).reshape(len(part), num_features, cfg.SEQUENCE_LENGTH, -1)

        return np.transpose(tensor, (3,1,0,2))


    
    def segment_annotation(self, data, start, end):
        """
        This function is to segment annotation activity by the date of experiment.
        """
        
        i_start = np.where(np.array(start > data.index[0]/1000)==True)[0][0]
        i_end = np.where(np.array(end < data.index[-1]/1000)==True)[0][-1]
        t_start = np.array(start)[i_start:i_end+1]
        t_end = np.array(end)[i_start:i_end+1]

        return t_start, t_end

    def segment_data(self, accel, gyro, start, end):
        """
        This function is to segment accel & gyro data by activity annotation.
        """
        
        i_start = np.where(gyro.index > 1000*start)[0][0]
        i_end = np.where(gyro.index < 1000*end)[0][-1]
        seg_gyro = gyro.loc[gyro.index[i_start : i_end], gyro.columns]
        if self.is_input:
            seg_accel = accel.loc[accel.index[i_start : i_end], accel.columns]
            seg_data = seg_gyro.join(seg_accel, how='outer')
        else:
            return seg_gyro

        return seg_data


    def resample_data(self, data):
        """
        This function is to resample data with given frequency.
        """
        
        re_freq = int((10**6)/cfg.SAMPLING_FREQ)
        data = data.loc[data.index[cfg.DATA_DROP+self.random_drop: -cfg.DATA_DROP], data.columns]
        data.index = data.index - data.index[0]

        overlap_index = data.index[data.index % re_freq == 0]
        new_index = [re_freq * i for i in range(int(data.index[-1]/re_freq)+1)]
        for i in range(len(overlap_index)):
            new_index.remove(overlap_index[i])
        new_index = pd.Index(new_index)
        new_index.name = data.index.name
        new_data = pd.DataFrame(index=new_index, columns=data.columns)
        
        data = data.append(new_data).sort_index(axis=0)
        data = data.interpolate(limit_area='inside')
        
        re_index = data.index % re_freq == 0
        data = data.loc[data.index[re_index], data.columns]
        if data.shape[0]%cfg.SEQUENCE_LENGTH > 0:
            data = data.loc[data.index[:-(data.shape[0]%cfg.SEQUENCE_LENGTH)], data.columns]

        return data


    def size_syncing_data(self, length, data):
        """
        This function is to sync each parts to have same size.
        """
        
        if self.is_input:
            part = self.input_part
        else:
            part = self.label_part

        reduce_length = length - np.array([length[:, i].min() for i in range(length.shape[1])])
        for p in part:
            for seq in range(reduce_length.shape[1]):
                index = part.index(p)
                accum = length[index][:seq+1].sum()
                if reduce_length[index][seq] != 0:
                    data[p] = data[p].drop(
                        data[p].index[accum - reduce_length[index][seq]:accum])
                    length[index][seq] = length[index][seq] - reduce_length[index][seq]
        return data



    def load_annotation(self, anno_dir):
        _, _, files = next(os.walk(anno_dir))
        anno_dir = os.path.join(anno_dir, files[0])
        annotation = pd.read_csv(anno_dir, index_col='AnnotationId')
        start = annotation['Start Timestamp (ms)']['Activity:Data Collection']
        end = annotation['Stop Timestamp (ms)']['Activity:Data Collection']

        return start, end


def build_IMU_data(is_train):
    
    return IMUData(is_train)