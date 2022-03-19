from extern_lib import *

import glob
import skimage.io as io
import skimage.transform as trans

seed = 42
np.random.seed = seed

DATA_PATH = 'training_data/'
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
# 数据加载
TRAIN_PATH = DATA_PATH + 'stage1_train/'  # 训练集路径
TEST_PATH = DATA_PATH + 'stage1_test/'  # 测试集路径


def plot_images(images):
    plt.clf()
    plt.figure(figsize=(20,20))
    for i in range(25):
        plt.subplot(5,5,i+1)  # 每行五列，共五行
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()

def load_data():
    train_ids = next(os.walk(TRAIN_PATH))[1]

    # 构造训练集输入和输出（mask）
    X_train = np.zeros((len(train_ids), IMG_HEIGHT,
                       IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT,
                       IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True)
        X_train[n] = img  # Fill empty X_train with values from img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        # mask
        # os.walk()文件、目录遍历器,在目录树中游走输出在目录中的文件名
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(
                mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)

        Y_train[n] = mask

    return X_train, Y_train


def load_test():
    test_ids = next(os.walk(TEST_PATH))[1]
    # 构造测试集输入
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,
                      IMG_CHANNELS), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True)
        X_test[n] = img
    return X_test
