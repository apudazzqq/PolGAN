import tensorflow as tf
import numpy as np
import cv2
from PanGan import PanGan
from DataSet import DataSet
from config import FLAGES
import os
# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# %%
if not os.path.exists(FLAGES.result_path):
    os.makedirs(FLAGES.result_path)
model = PanGan(FLAGES.pan_size, FLAGES.ms_size, 1, FLAGES.ratio, 0.001, 0.99, 1000, is_training=False)
saver = tf.train.Saver()
dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, './data/test/test_qk.h5', 1, FLAGES.stride)
# %%
h = 16369
w = 2740
img = np.zeros((h, w), dtype='float32')
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGES.model_path)
    # ms_test_path = FLAGES.test_path + '/lrms'
    # pan_test_path = FLAGES.test_path + '/pan'
    # for test_itr in range(dataset.pan.shape[0]):
    test_itr = 0
    for i in range(0, h - FLAGES.pan_size, FLAGES.stride * 64):
        for j in range(0, w - FLAGES.pan_size, FLAGES.stride * 64):
            if (i + FLAGES.pan_size <= h) & (j + FLAGES.pan_size <= w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[i:i + FLAGES.pan_size, j:j + FLAGES.pan_size] = PanSharpening
                save_name = str(test_itr).zfill(4) + '.tif'
                cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size > h) & (j + FLAGES.pan_size <= w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[h - FLAGES.pan_size:, j:j + FLAGES.pan_size] = PanSharpening
                save_name = str(test_itr).zfill(4) + '.tif'
                cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size <= h) & (j + FLAGES.pan_size > w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[i:i + FLAGES.pan_size, w - FLAGES.pan_size:] = PanSharpening
                save_name = str(test_itr).zfill(4) + '.tif'
                cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size > h) & (j + FLAGES.pan_size > w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[h - FLAGES.pan_size:, w - FLAGES.pan_size:] = PanSharpening
                save_name = str(test_itr).zfill(4) + '.tif'
                cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGES.model_path)
    # ms_test_path = FLAGES.test_path + '/lrms'
    # pan_test_path = FLAGES.test_path + '/pan'
    # for test_itr in range(dataset.pan.shape[0]):
    test_itr = 0
    for i in range(0, h - FLAGES.pan_size, FLAGES.stride * 64):
        for j in range(0, w - FLAGES.pan_size, FLAGES.stride * 64):
            if (i + FLAGES.pan_size <= h) & (j + FLAGES.pan_size <= w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[i + 128:i + FLAGES.pan_size, j + 128:j + FLAGES.pan_size] = PanSharpening[128:FLAGES.pan_size, 128:FLAGES.pan_size]
                save_name = str(test_itr).zfill(4) + '.tif'
                # cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size > h) & (j + FLAGES.pan_size <= w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[h - 384:, j + 128:j + FLAGES.pan_size] = PanSharpening[128:FLAGES.pan_size, 128:FLAGES.pan_size]
                save_name = str(test_itr).zfill(4) + '.tif'
                # cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size <= h) & (j + FLAGES.pan_size > w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[i + 128:i + FLAGES.pan_size, w - 384:] = PanSharpening[128:FLAGES.pan_size, 128:FLAGES.pan_size]
                save_name = str(test_itr).zfill(4) + '.tif'
                # cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

            if (i + FLAGES.pan_size > h) & (j + FLAGES.pan_size > w):
                pan = dataset.pan[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                ms = dataset.ms[test_itr].reshape(1, FLAGES.ms_size, FLAGES.ms_size, 1)
                height = dataset.height[test_itr].reshape(1, FLAGES.pan_size, FLAGES.pan_size, 1)
                PanSharpening, error = sess.run([model.PanSharpening_img, model.g_spatial_loss],
                                                feed_dict={model.pan_img: pan, model.ms_img: ms,
                                                           model.height_img: height})
                PanSharpening = PanSharpening * 20 + 20
                PanSharpening = PanSharpening.squeeze()
                PanSharpening = PanSharpening.astype('float32')
                img[h - 384:, w - 384:] = PanSharpening[128:FLAGES.pan_size, 128:FLAGES.pan_size]
                save_name = str(test_itr).zfill(4) + '.tif'
                # cv2.imwrite(os.path.join(FLAGES.result_path, save_name), PanSharpening)
                print(str(test_itr) + ' done.' + 'spatial error is ' + str(error))
                test_itr = test_itr + 1

cv2.imwrite(os.path.join(FLAGES.result_path, 'howland_pangan_spatialdloss_spatialgloss.tif'), img)