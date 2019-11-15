'''
This file is to set default configuration of the model.
TODO : Generate argparser or yaml file to change default value
'''

class cfg(object):
    VERSION = 1

    # Model define
    ARCHITECTURE = "ResNet"
    MODEL_DEVICE = "cuda:0"
    #NUM_LAYER = 2
    REF_CHANNEL = 32

    # Dataset define
    USE_PKL = True
    DATASET = 'PanopticDome'
    IMU_DIR = '../Dataset/synced_IMU'
    KEYPOINTS_DIR = '../Dataset/keypoints'
    PKL_DIR = '../Dataset/PKL/head/'
    TENSOR_TYPE = "torchFloatTensor"
    DATA_DEVICE = "cuda:0"

    # Train config
    DATA_DATE = ['190503', '190510', '190517', '190607']
    LOAD_ALL = False
    IMU_SET = [1, 2]

    KEYPOINT_PART = [1]
    
    TRAIN_PART = ['head']
    RANDOM_SAMPLING = False
    DATA_DROP = 1000
    SEQUENCE_LENGTH = 25
    SAMPLING_FREQ = 125
    NUM_EPOCH = 100
    LEARNING_RATE = 0.00125
    BATCH_SIZE = 8

    OUTPUT_DIR = '../output'
    LOG_DIR = '../output/log'
    CHECKPOINT = 25000
    PRETRAINED = False
    PRETRAINED_MODEL = '00100000.pt'

    UNSYNCED_LIST = ['190510_2_1', '190517_11_1', '190517_1_2', '190517_11_2', 
        '190517_8_1', '190517_8_2', '190517_12_2', '190607_4_2', '190607_8_1', '190607_8_2', '190607_11_1', '190607_11_2']