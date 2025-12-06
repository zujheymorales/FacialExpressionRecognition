import os
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers


# path to directories for data 
BASE_DIR = "data/fer/images/images"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val") 


