import tensorflow as tf
import numpy as np
import ops as Op
from PSGAN_lib import conv, lrelu, batchnorm, strided_conv


class PanGan(object):

    def __init__(self, pan_size, ms_size, batch_size, ratio, init_lr=0.001, lr_decay_rate=0.99, lr_decay_step=1000,
                 is_training=True):

        self.is_training = is_training
        self.ratio = ratio
        self.batch_size = batch_size
        self.pan_size = pan_size
        self.ms_size = ms_size
        self.init_lr = init_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.build_model(pan_size, ms_size, batch_size, is_training)

    def build_model(self, pan_size, ms_size, batch_size, is_training):

        if is_training:
            with tf.name_scope('input'):
                self.pan_img = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1),
                                                        name='pan_placeholder')  # HR
                self.height_img = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1),
                                                           name='height_placeholder')  # HR
                self.ms_img = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, ms_size, ms_size, 1),
                                                       name='ms_placeholder')  # LR
                self.pan_img_hp = self.high_pass(self.pan_img)  # HR

            with tf.name_scope('PanSharpening'):
                self.PanSharpening_img = self.PanSharpening_model_dense(self.pan_img, self.ms_img)  # HR
                # self.PanSharpening_img_blur=self.blur(self.PanSharpening_img,7)
                self.PanSharpening_img_ = tf.image.resize_images(images=self.PanSharpening_img, size=[ms_size, ms_size],
                                                                 method=tf.image.ResizeMethod.BILINEAR)  # LR
                self.PanSharpening_img_pan = tf.reshape(self.PanSharpening_img, (batch_size, pan_size, pan_size, 1))
                #self.PanSharpening_img_ms = tf.image.resize_images(self.PanSharpening_img, [batch_size, ms_size, ms_size, 1], method=2)
                self.PanSharpening_img_hp = self.high_pass(self.PanSharpening_img_pan)

            with tf.name_scope('d_loss'):
                with tf.name_scope('spatial_loss'):
                    spatial_pos = self.spatial_discriminator(self.pan_img, reuse=False)
                    spatial_neg = self.spatial_discriminator(self.PanSharpening_img_pan, reuse=True)
                    spatial_pos_loss = tf.reduce_mean(
                        tf.square(spatial_pos - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
                    spatial_neg_loss = tf.reduce_mean(
                        tf.square(spatial_neg - tf.zeros(shape=[batch_size, 1], dtype=tf.float32)))
                    coherence_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_pan*20+20 - self.height_img))
                    #self.spatial_loss = spatial_pos_loss + spatial_neg_loss + coherence_loss
                    self.spatial_loss = spatial_pos_loss + spatial_neg_loss
                    tf.summary.scalar('spatial_loss', self.spatial_loss)

            with tf.name_scope('g_loss'):
                spatial_loss_ad = tf.reduce_mean(tf.square(spatial_neg - tf.ones(shape=[batch_size, 1], dtype=tf.float32)))
                tf.summary.scalar('spatial_loss_ad', spatial_loss_ad)
                g_spatital_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_hp - self.pan_img_hp))
                tf.summary.scalar('g_spatital_loss', g_spatital_loss)
                #height_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_ms - self.ms_img))
                height_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_pan * 20 + 20 - self.height_img))
                #self.g_loss = spatial_loss_ad + g_spatital_loss + height_loss
                self.g_loss = spatial_loss_ad + g_spatital_loss
                tf.summary.scalar('g_loss', self.g_loss)

            # with tf.name_scope('valid_error'):
            # self.valid_spatital_error=tf.reduce_mean(tf.abs(self.PanSharpening_img_hp-self.pan_img_hp))
            # tf.summary.scalar('valid_spatital_error', self.valid_spatital_error)
            # self.valid_spectrum_error=tf.reduce_mean(tf.abs(self.PanSharpening_img-self.ms_img_))
            # tf.summary.scalar('valid_spectrum_error', self.valid_spectrum_error)
        else:
            with tf.name_scope('input'):
                self.pan_img = tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1),
                                              name='pan_placeholder')
                self.height_img = tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1),
                                                 name='height_placeholder')  # HR
                self.ms_img = tf.placeholder(dtype=tf.float32, shape=(batch_size, ms_size, ms_size, 1),
                                             name='ms_placeholder')
            self.PanSharpening_img = self.PanSharpening_model_dense(self.pan_img, self.ms_img)
            self.PanSharpening_img_pan = tf.reshape(self.PanSharpening_img, (batch_size, pan_size, pan_size, 1))
            # self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[ms_size, ms_size], method=tf.image.ResizeMethod.BILINEAR)
            # PanSharpening_img_hp=self.high_pass(self.PanSharpening_img)
            # pan_img_hp=self.high_pass(self.pan_img)
            self.g_spatial_loss = tf.reduce_mean(tf.square(self.PanSharpening_img_pan*20+20 - self.height_img))
            # self.g_spatial_loss=tf.reduce_mean(tf.square(PanSharpening_img_hp-pan_img_hp))

    def train(self):
        t_vars = tf.trainable_variables()
        d_spatial_vars = [var for var in t_vars if 'spatial_discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Pan_model' in var.name]
        with tf.name_scope('train_step'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(self.init_lr, global_step=self.global_step,
                                                            decay_rate=self.lr_decay_rate,
                                                            decay_steps=self.lr_decay_step)
            tf.summary.scalar('global learning rate', self.learning_rate)
            self.train_Pan_model = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars,
                                                                                          global_step=self.global_step)
            self.train_spatial_discrim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.spatial_loss,
                                                                                                var_list=d_spatial_vars)

    def PanSharpening_model_dense(self, pan_img, ms_img):
        with tf.compat.v1.variable_scope('Pan_model'):
            # if self.is_training:
            with tf.name_scope('upscale'):
                ms_img = tf.image.resize_images(ms_img, [512, 512], method=2)
            input = tf.concat([ms_img, pan_img], axis=-1)
            with tf.compat.v1.variable_scope('layer1'):
                weights = tf.compat.v1.get_variable("w1", [9, 9, 1 + 1, 64],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
                conv1 = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                    decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.relu(conv1)
            with tf.compat.v1.variable_scope('layer2'):
                weights = tf.compat.v1.get_variable("w2", [5, 5, 64 + 1 + 1, 32],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
                conv2 = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(tf.concat([input, conv1], -1), weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                    decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv2 = tf.nn.relu(conv2)
            with tf.compat.v1.variable_scope('layer3'):
                weights = tf.compat.v1.get_variable("w3", [5, 5, 1 + 1 + 64 + 32, 1],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b3", [1], initializer=tf.constant_initializer(0.0))
                conv3 = (tf.nn.conv2d(tf.concat([input, conv1, conv2], -1), weights, strides=[1, 1, 1, 1],
                                      padding='SAME') + bias)

                conv3 = tf.tanh(conv3)

        return conv3

    def PSGAN_model(self, pan_img, height_ms_img):
        with tf.variable_scope('PSGAN_model'):
            with tf.name_scope('upscale'):
                height_ms_img = tf.image.resize_images(height_ms_img, [512, 512], method=2)
            layers = []  # 特征提取网络
            with tf.variable_scope("encoder_1_1"):  # 分支1
                output = conv(pan_img, 3, 32, 1)  # 3*3*32/1   128*128*32
                layers.append(output)
            with tf.variable_scope("encoder_2_1"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 32, 1)  # 3*3*32/1   128*128*32
                layers.append(convolved)
            with tf.variable_scope("encoder_3_1"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 2, 64, 2)  # 2*2*64/2   64*64*64
                layers.append(convolved)

            with tf.variable_scope("encoder_1_2"):  # 分支2
                output = conv(height_ms_img, 3, 32, 1)  # 3*3*32/1   128*128*32
                layers.append(output)
            with tf.variable_scope("encoder_2_2"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 32, 1)  # 3*3*32/1   128*128*32
                layers.append(convolved)
            with tf.variable_scope("encoder_3_2"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 2, 64, 2)  # 2*2*64/2   64*64*64
                layers.append(convolved)
            # 特征融合网络
            concat1 = tf.concat([layers[-1], layers[-1 - 3]], 3)  # 拼接两个tensor, 在维度3上（channel）64*64*128
            with tf.variable_scope("encoder_4"):
                rectified = lrelu(concat1, 0.2)
                convolved = conv(rectified, 3, 128, 1)  # 3*3*128/1  64*64*128
                layers.append(convolved)
            with tf.variable_scope("encoder_5"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 128, 1)  # 3*3*128/1  64*64*128
                layers.append(convolved)
            with tf.variable_scope("encoder_6"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 256, 2)  # 3*3*256/2  32*32*256
                layers.append(convolved)
            # 图像重建网络
            with tf.variable_scope("decoder_7"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 1, 256, 1)  # 1*1*256/1  32*32*256
                layers.append(convolved)

            with tf.variable_scope("decoder_8"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 256, 1)  # 3*3*256/1  32*32*256
                layers.append(convolved)

            with tf.variable_scope("decoder_9"):
                rectified = lrelu(layers[-1], 0.2)
                strided_convolved = strided_conv(rectified, 2, 128)  # 2*2*128/2     64*64*128
                layers.append(strided_convolved)

            concat2 = tf.concat([layers[-1], layers[-1 - 4]], 3)  # 64*64*256

            with tf.variable_scope("decoder_10"):
                rectified = lrelu(concat2, 0.2)
                convolved = conv(rectified, 3, 128, 1)  # 3*3*128/1     64*64*128
                layers.append(convolved)

            with tf.variable_scope("decoder_11"):
                rectified = lrelu(layers[-1], 0.2)
                strided_convolved = strided_conv(rectified, 2, 128)  # 2*2*128/2     128*128*128
                layers.append(strided_convolved)

            concat3 = tf.concat([layers[-1], layers[-1 - 9], layers[-1 - 12]], 3)  # 128*128*192

            with tf.variable_scope("decoder_12"):
                rectified = lrelu(concat3, 0.2)
                convolved = conv(rectified, 3, 64, 1)  # 3*3*64/1  128*128*64
                layers.append(convolved)

            with tf.variable_scope("decoder_13"):
                rectified = lrelu(layers[-1], 0.2)
                convolved = conv(rectified, 3, 1, 1)  # 3*3*4/1 128*128*4
                output = tf.tanh(convolved)
                layers.append(output)

        return layers[-1]

    def spatial_discriminator(self, img_hp, reuse=False):
        with tf.compat.v1.variable_scope('spatial_discriminator', reuse=reuse):
            with tf.compat.v1.variable_scope('layer_1'):
                weights = tf.compat.v1.get_variable("w_1", [3, 3, 1, 16],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_1", [16], initializer=tf.constant_initializer(0.0))
                conv1_spatial = tf.nn.conv2d(img_hp, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
                conv1_spatial = self.lrelu(conv1_spatial)
                print(conv1_spatial.shape)
            with tf.compat.v1.variable_scope('layer_2'):
                weights = tf.compat.v1.get_variable("w_2", [3, 3, 16, 32],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_2", [32], initializer=tf.constant_initializer(0.0))
                conv2_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv1_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv2_spatial = self.lrelu(conv2_spatial)
                print(conv2_spatial.shape)
            with tf.compat.v1.variable_scope('layer_3'):
                weights = tf.compat.v1.get_variable("w_3", [3, 3, 32, 64],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_3", [64], initializer=tf.constant_initializer(0.0))
                conv3_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv2_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv3_spatial = self.lrelu(conv3_spatial)
                print(conv3_spatial.shape)
            with tf.compat.v1.variable_scope('layer_4'):
                weights = tf.compat.v1.get_variable("w_4", [3, 3, 64, 128],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_4", [128], initializer=tf.constant_initializer(0.0))
                conv4_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv3_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv4_spatial = self.lrelu(conv4_spatial)
                print(conv4_spatial.shape)
                # conv4_spatial = tf.reshape(conv4_spatial, [self.batch_size, 4 * 4 * 128])
            with tf.compat.v1.variable_scope('layer_5'):
                weights = tf.compat.v1.get_variable("w_5", [3, 3, 128, 256],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_5", [256], initializer=tf.constant_initializer(0.0))
                conv5_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv4_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv5_spatial = self.lrelu(conv5_spatial)
                print(conv5_spatial.shape)

            with tf.compat.v1.variable_scope('layer_6'):
                weights = tf.compat.v1.get_variable("w_6", [3, 3, 256, 512],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_6", [512], initializer=tf.constant_initializer(0.0))
                conv6_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv5_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv6_spatial = self.lrelu(conv6_spatial)
                print(conv6_spatial.shape)

            with tf.compat.v1.variable_scope('line_7'):
                weights = tf.compat.v1.get_variable("w_7", [8, 8, 512, 1],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b_7", [1], initializer=tf.constant_initializer(0.0))
                conv7_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv6_spatial, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                print(conv7_spatial.shape)
                conv7_spatial = self.lrelu(conv7_spatial)
                conv7_spatial = tf.reshape(conv7_spatial, [self.batch_size, 1])

                # line5_spatial = tf.matmul(conv4_spatial, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return conv7_spatial

    def high_pass(self, img):

        blur_kerel = np.zeros(shape=(3, 3, 1, 1), dtype=np.float32)
        value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        blur_kerel[:, :, 0, 0] = value
        img_hp = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')
        # img_hp=tf.reshape(tf.reduce_mean(img_hp,3),[self.batch_size,128,128,1])
        # img_hp=img-img_lp
        return img_hp

    def blur(self, img, kernel_size, gaussian_variance=1):
        blur_kerel = np.zeros(shape=(kernel_size, kernel_size, 1, 1), dtype=np.float32)
        value = self.generate_Gauss(kernel_size)
        for i in range(1):
            blur_kerel[:, :, i, i] = value
        img_blur = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')
        return img_blur

    def generate_Gauss(self, kernel_size, gaussian_variance=1):
        x = np.arange(0, kernel_size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = kernel_size // 2
        kernel = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / gaussian_variance / gaussian_variance)
        kernel = kernel / np.sum(kernel)
        return kernel

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)