import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
