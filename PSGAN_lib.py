import tensorflow as tf

def conv(batch_input, kernel_size, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        #padded_input = tf.pad(batch_input,[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
        #conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        conv = tf.nn.conv2d(batch_input, filter, [1,stride, stride, 1], padding='SAME')
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x=tf.identity(x) # 转换成tensor
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset",[channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0,0.02))
        mean, variance = tf.nn.moments(input,axes=[0,1,2], keep_dims=False) # 计算均值和方差
        variance_epsilon= 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def strided_conv(batch_input, kernel_size, out_channels):
    with tf.variable_scope("strided_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        strided_conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height*2, in_width*2, out_channels], [1,2,2,1], padding='SAME') # 转置卷积，图像的宽和高都翻倍
        return strided_conv