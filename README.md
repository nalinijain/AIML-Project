---
This code is only tested on following env.\
\
Python version = 3.7.3\
PyTorch version = 1.2.0\
CUDA version = 10.1




# Data Structure
Current code expect that data is located at Dataset/"Subject id (ex:Set01)"/"Date (YYMMDD)"/"Part (ex: head, Lshank)"/*.csv




# Configuration Setting
Default configuration is defined at model/defaults.py.




# Neural Network Model
I coded based on pyTorch framework. The model class is at model/torch_model.py and specified blocks or layers are defined at model/torch_layer.py. If you developed new model, then add it on torch_model.py and if you developed specific layers (activation function / pooling layer / conv layer) add it on torch_layer.py.




# Run the code
You can run the code by command 
```
python ./model/trainer.py
```
I encourage you to debug it using pdb to understand the code and feel free to change or remove the lines if necessary.




# TODO:
1) build loss function
2) build evaluator (test the model)
3) build save and load the model
4) ...etc
