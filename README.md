---
This code is only tested on following env.\
\
Python version = 3.7.3\
PyTorch version = 1.2.0\
CUDA version = 10.1




# Data Structure
Directory of loading PKL file is defined at defualts.py. Download PKL file from google drive and make sure to match the directory of PKL file with PKL_DIR in defaults.py.




# Configuration Setting
Default configuration is defined at model/defaults.py.




# Neural Network Model
I coded based on pyTorch framework. The model class is at model/torch_model.py and specified blocks or layers are defined at model/torch_layer.py. If you developed new model, then add it on torch_model.py and if you developed specific layers (activation function / pooling layer / conv layer) add it on torch_layer.py.




# Run the code
You can run the code by command 
```
cd model/ && python trainer.py
```
I encourage you to debug it using pdb to understand the code and feel free to change or remove the lines if necessary.




# TODO:
1) build evaluator (test the model)
2) ...etc
