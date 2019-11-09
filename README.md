---
This code is only tested on following env.\
\
Python version = 3.7.3\
PyTorch version = 1.2.0\
CUDA version = 10.1

# Pre-requisite:
1) Framework : PyTorch and Python
2) Libraries : Numpy, Pickle, Pandas, Tensorboard


# Data Structure
Current code expect that data is located at Dataset/"Subject id (ex:Set01)"/"Date (YYMMDD)"/"Part (ex: head, Lshank)"/*.csv
Please place your .pkl file at Dataset folder. I recommend you to make a soft link.




# Configuration Setting
Default configuration is defined at model/defaults.py. 
1. In this file, you can choose the architecture of neural network. At this point, two networks (CNN and RNN : bi-directional LSTM) has been implemented and you can set ARCHITECTURE = "ResNet" for CNN, ARCHITECTURE = "LSTM" for RNN.
2. You can choose whether to load data from csv file or pkl file. In defaults.py USE_PKL, True for using PKL file and False for loading data from csv file.
3. For loading data from csv file, you can define the length of sequence (SEQUENCE_LENGTH), frequence of data (SAMPLING_FREQ), how many data from the beginning and the end to delete (DATA_DROP).
4. You can set your training parameters (Total number of epoch, Learning rate, Batch size) either.
5. If you have saved your model at output/ directory, and if you want to resume your training, you can set PRETRAINED as True, and put the name of saved model .pt file for PRETRAINED_MODEL.




# Neural Network Model
I coded based on pyTorch framework. The model class is at model/torch_model.py and specified blocks or layers are defined at model/torch_layer.py. If you developed new model, then add it on torch_model.py and if you developed specific layers (activation function / pooling layer / conv layer) add it on torch_layer.py.




# Run the code
You can run the code by command 
```
python ./model/trainer.py
```
I encourage you to debug it using pdb to understand the code and feel free to change or remove the lines if necessary.




# TODO:
Let me think...
