#%%
import numpy as np
import os
import h5py
import scipy.io as scio
#%%
class DataSet(object):
    def __init__(self, pan_size, ms_size, source_path, data_save_path, batch_size, stride, category='train'):
        self.pan_size = pan_size
        self.ms_size = ms_size
        self.batch_size = batch_size
        if not os.path.exists(data_save_path):
            self.make_data(source_path, data_save_path, stride)
        self.pan, self.ms, self.height = self.read_data(data_save_path, category)
        self.data_generator = self.generator()

    def generator(self):
        num_data = self.pan.shape[0]
        while True:
            batch_pan = np.zeros((self.batch_size, self.pan_size, self.pan_size, 1))
            batch_ms = np.zeros((self.batch_size, self.ms_size, self.ms_size, 1))
            batch_height = np.zeros((self.batch_size, self.pan_size, self.pan_size, 1))
            for i in range(self.batch_size):
                random_index = np.random.randint(0, num_data)
                batch_pan[i] = self.pan[random_index]
                batch_ms[i] = self.ms[random_index]
                batch_height[i] = self.height[random_index]
            yield batch_pan, batch_ms, batch_height

    def read_data(self, path, category):
        f = h5py.File(path, 'r')
        if category == 'train':
            pan = np.array(f['pan_train'])
            ms = np.array(f['ms_train'])
            height = np.array(f['height_train'])
        else:
            pan = np.array(f['pan_train'])
            ms = np.array(f['ms_train'])
            height = np.array(f['height_train'])
        return pan, ms, height

    def make_data(self, source_path, data_save_path, stride):
        source_ms_path = os.path.join(source_path, 'HeightMS', 'HeightMS.mat')
        source_pan_path = os.path.join(source_path, 'Pan', 'PauliPan.mat')
        source_height_path = os.path.join(source_path, 'HeightPan', 'HeightPan.mat')
        pan_train = self.crop_to_patch(source_pan_path, stride, name='pan')
        ms_train = self.crop_to_patch(source_ms_path, stride, name='ms')
        height_train = self.crop_to_patch(source_height_path, stride, name='height')
        print('The number of ms patch is: ' + str(len(ms_train)))
        print('The number of pan patch is: ' + str(len(pan_train)))
        print('The number of height patch is: ' + str(len(height_train)))
        f = h5py.File(data_save_path, 'w')
        f.create_dataset('pan_train', data=pan_train)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('height_train', data=height_train)

    def crop_to_patch(self, img_path, stride, name):
        img = self.read_img2(img_path)
        h = img.shape[0]
        w = img.shape[1]
        print(h)
        print(w)
        count=0
        all_img = []
        if name == 'ms':
            for i in range(0, h - self.ms_size, stride):
                for j in range(0, w - self.ms_size, stride):
                    if (i + self.ms_size <= h) & (j + self.ms_size <= w):
                        img_patch = img[i:i + self.ms_size, j:j + self.ms_size].reshape(self.ms_size,self.ms_size,1)
                        all_img.append(img_patch)
                        count = count+1
                    if (i + self.ms_size > h) & (j + self.ms_size <= w):
                        img_patch = img[h - self.ms_size:, j:j + self.ms_size].reshape(self.ms_size,self.ms_size,1)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.ms_size <= h) & (j + self.ms_size > w):
                        img_patch = img[i:i + self.ms_size, w - self.ms_size:].reshape(self.ms_size, self.ms_size, 1)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.ms_size > h) & (j + self.ms_size > w):
                        img_patch = img[h - self.ms_size:, w - self.ms_size:].reshape(self.ms_size, self.ms_size, 1)
                        all_img.append(img_patch)
                        count = count + 1

        else:
            for i in range(0, h - self.pan_size, stride * 64):
                for j in range(0, w - self.pan_size, stride * 64):
                    if (i + self.pan_size <= h) & (j + self.pan_size <= w):
                        img_patch = img[i:i + self.pan_size, j:j + self.pan_size].reshape(self.pan_size,self.pan_size,1)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size > h) & (j + self.pan_size <= w):
                        img_patch = img[h - self.pan_size:, j:j + self.pan_size].reshape(self.pan_size,self.pan_size,1)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size <= h) & (j + self.pan_size > w):
                        img_patch = img[i:i + self.pan_size, w - self.pan_size:].reshape(self.pan_size, self.pan_size, 1)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size > h) & (j + self.pan_size > w):
                        img_patch = img[h - self.pan_size:, w - self.pan_size:].reshape(self.pan_size, self.pan_size, 1)
                        all_img.append(img_patch)
                        count = count + 1
        return all_img

    def crop_to_patch_rgb(self, img_path, stride, name):
        img = self.read_img2(img_path)
        h = img.shape[0]
        w = img.shape[1]
        d = img.shape[2]
        print(h)
        print(w)
        print(d)
        count = 0
        all_img = []
        if name == 'ms':
            for i in range(0, h - self.ms_size, stride):
                for j in range(0, w - self.ms_size, stride):
                    if (i + self.ms_size <= h) & (j + self.ms_size <= w):
                        img_patch = img[i:i + self.ms_size, j:j + self.ms_size, :].reshape(self.ms_size, self.ms_size, d)
                        all_img.append(img_patch)
                        count = count+1
                    if (i + self.ms_size > h) & (j + self.ms_size <= w):
                        img_patch = img[h - self.ms_size:, j:j + self.ms_size, :].reshape(self.ms_size, self.ms_size, d)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.ms_size <= h) & (j + self.ms_size > w):
                        img_patch = img[i:i + self.ms_size, w - self.ms_size:, :].reshape(self.ms_size, self.ms_size, d)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.ms_size > h) & (j + self.ms_size > w):
                        img_patch = img[h - self.ms_size:, w - self.ms_size:, :].reshape(self.ms_size, self.ms_size, d)
                        all_img.append(img_patch)
                        count = count + 1

        else:
            for i in range(0, h - self.pan_size, stride * 64):
                for j in range(0, w - self.pan_size, stride * 64):
                    if (i + self.pan_size <= h) & (j + self.pan_size <= w):
                        img_patch = img[i:i + self.pan_size, j:j + self.pan_size, :].reshape(self.pan_size, self.pan_size, d)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size > h) & (j + self.pan_size <= w):
                        img_patch = img[h - self.pan_size:, j:j + self.pan_size, :].reshape(self.pan_size, self.pan_size, d)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size <= h) & (j + self.pan_size > w):
                        img_patch = img[i:i + self.pan_size, w - self.pan_size:, :].reshape(self.pan_size, self.pan_size, d)
                        all_img.append(img_patch)
                        count = count + 1
                    if (i + self.pan_size > h) & (j + self.pan_size > w):
                        img_patch = img[h - self.pan_size:, w - self.pan_size:, :].reshape(self.pan_size, self.pan_size, d)
                        all_img.append(img_patch)
                        count = count + 1
        return all_img

    def read_img2(self, path):
        img = scio.loadmat(path)['I']
        # img = (img - 127.5) / 127.5

        return img









