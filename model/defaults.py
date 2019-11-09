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
    IMU_DIR = '../Dataset/IMU'
    KEYPOINTS_DIR = '../Dataset/keypoints'
    TENSOR_TYPE = "torchFloatTensor"
    DATA_DEVICE = "cuda:0"

    # Train config
    IMU_SET = [1, 2]
    TRAIN_PART = ['head', 'Lfoot', 'Rfoot']
    TRAIN_LABEL_PART = ['Lfoot', 'Rfoot'] 
    TEST_PART = ['head']
    DATA_DATE = ['190503', '190510', '190517', '190531', '190607']
    RANDOM_SAMPLING = True
    DATA_DROP = 1000
    SEQUENCE_LENGTH = 125
    SAMPLING_FREQ = 125
    NUM_EPOCH = 1000
    LEARNING_RATE = 0.00125
    BATCH_SIZE = 8

    OUTPUT_DIR = '../output'
    LOG_DIR = '../output/log'
    CHECKPOINT = 25000
    PRETRAINED = False