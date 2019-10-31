'''
This file is to set default configuration of the model.
TODO : Generate argparser or yaml file to change default value
'''
import wandb

class cfg(object):
    VERSION = 1

    # Model define
    ARCHITECTURE = "ResNet"
    MODEL_DEVICE = "cuda:0"
    #NUM_LAYER = 2
    REF_CHANNEL = 32

    # Dataset define
    DATASET = 'PanopticDome'
    IMU_DIR = '../Dataset'
    KEYPOINTS_DIR = '../Dataset/keypoints'
    TENSOR_TYPE = "torchFloatTensor"
    DATA_DEVICE = "cuda:0"

    # Train config
    LOAD_ALL = False
    IMU_SET = [1]       # [1, 2]    
    TRAIN_PART = ['head']
    TRAIN_LABEL_PART = ['Rshank', 'Lshank'] 
    TEST_PART = ['head']
    DATA_DATE = ['190607']
    RANDOM_SAMPLING = True
    wandb.config.DATA_DROP = 1000
    wandb.config.SEQUENCE_LENGTH = 128
    wandb.config.SAMPLING_FREQ = 125
    wandb.config.NUM_EPOCH = 1000
    wandb.config.LEARNING_RATE = 0.00125
    wandb.config.BATCH_SIZE = 8
