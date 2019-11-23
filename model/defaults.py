'''
This file is to set default configuration of the model.
TODO : Generate argparser or yaml file to change default value
'''

class cfg(object):
    VERSION = 1

    # Model define
    ARCHITECTURE = "ResNet"
    MODEL_DEVICE = "cuda:0"
    #MODEL_DEVICE = "cpu"
    #NUM_LAYER = 2
    REF_CHANNEL = 32

    # Dataset define
    USE_PKL = True
    DATASET = 'PanopticDome'
    IMU_DIR = '../Dataset/synced_IMU'
    KEYPOINTS_DIR = '../Dataset/keypoints'
    PKL_DIR = '../Dataset/PKL/'
    TENSOR_TYPE = "torchFloatTensor"
    DATA_DEVICE = "cuda:0"
    DATA_DATE = ['190503', '190510', '190517', '190607']
    LOAD_ALL = False
    IMU_SET = [1, 2]
    TRAIN_PART = ['head']
    KEYPOINT_PART = [1]
    DATA_AUGMENTATION = False
    NUM_INIT_POSES = 1
    RANDOM_SAMPLING = False
    DATA_DROP = 1000
    SEQUENCE_LENGTH = 25
    SAMPLING_FREQ = 125
    UNSYNCED_LIST = ['190510_2_1', '190517_11_1',  '190517_14_1','190517_1_2', '190517_11_2', '190517_14_2',
        '190517_8_1', '190517_8_2', '190517_12_2', '190607_4_2', '190607_8_1', '190607_8_2', '190607_11_1', '190607_11_2']
    

    # Train Config
    NUM_EPOCH = 100
    LEARNING_RATE = 0.00125
    BATCH_SIZE = 32
    CONSIDER_INIT_CONF = True
    L2_LW = 1
    
    
    OUTPUT_DIR = '../output'
    LOG_DIR = '../output/log'
    CHECKPOINT = 25000
    PRETRAINED = False
    PRETRAINED_MODEL = '00100000.pt'