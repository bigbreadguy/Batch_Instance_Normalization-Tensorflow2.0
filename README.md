# Batch_Instance_Normalization-Tensorflow2.0
Simple Tensorflow 2.0 implementation of [Batch-Instance Normalization (NIPS 2018)](https://arxiv.org/abs/1805.07925).<br/>
Originally implemented in Tensorflow version 1 by [taki0112](https://github.com/taki0112/Batch_Instance_Normalization-Tensorflow).<br/>
Edited by Seunggeon Lim in reference to InstanceNormalization layer from [Tensorflow examples pix2pix.py](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)

# Usage
**Sequential API**
```
model = tf.keras.Sequential()
model.add(BatchInstanceNormalization())
```
**Functional API**
```
conv = tf.keras.layers.Conv2D()(input)
norm = BatchInstanceNormalization()(conv)
```
