import tensorflow as tf

class BatchInstanceNormalization(tf.keras.layers.Layer):
    """Batch Instance Normalization Layer (https://arxiv.org/abs/1805.07925)."""

    def __init__(self, epsilon=1e-5):
        super(BatchInstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)
    
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',        
            trainable=True)
    
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
  
    def call(self, x):
        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) * (tf.math.rsqrt(batch_sigma + self.epsilon))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) * (tf.math.rsqrt(ins_sigma + self.epsilon))

        return (self.rho * x_batch + (1 - self.rho) * x_ins) * self.gamma + self.beta
