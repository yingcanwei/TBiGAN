#encoding=utf8

import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os, gzip
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.image as mp

def load_mnist(seed,dataset_name):
    print('loading data')

    data_dir = os.path.join("../", dataset_name)

    data = np.load(data_dir)

    # combine the x_train and x_v

    if dataset_name == 'mnist.npz':
        train = data['x_train'].astype(np.float32)
    elif dataset_name == 'mnist_r.npz':
        train = data['train'].reshape((-1,784)).astype(np.float32)
    else:
        train = data['train'].reshape((-1,2352)).astype(np.float32)
    rng = np.random.RandomState(seed)

    #train = train[rng.permutation(train.shape[0])]  # shuffling unl dataset

    return train

def load_meta(dataset_name):
    print('loading data')

    data_dir = os.path.join("../", dataset_name)

    data = np.load(data_dir)

    coder = data['Dcoder']
    recons = data['Dlike']



    #train = train[rng.permutation(train.shape[0])]  # shuffling unl dataset
    return recons, coder


def load_mnist_3(seed,dataset_name):
    print('loading data')

    data_dir = os.path.join("../", dataset_name)

    data = np.load(data_dir)

    # combine the x_train and x_vs

    if dataset_name == 'mnist.npz' or dataset_name == 'mnist_r.npz' or dataset_name == 'usps.npz':
        train = data['x_train'].reshape((-1,28,28,1)).astype(np.float32)
        train = np.concatenate([train, train, train], 3).reshape(-1,2352)
    elif (dataset_name == 'SVHN.npz'):
        train = data['x_train'].reshape((-1, 2352)).astype(np.float32)
    elif ('PCA' in dataset_name ):
        train = data['z'].astype(np.float32)
    elif ('Coder' in dataset_name ):
        train = data['Dcoder'].astype(np.float32)
    else:
        train = data['train'].reshape((-1,2352)).astype(np.float32)

    #train = train[rng.permutation(train.shape[0])]  # shuffling unl dataset

    return train

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    #return scipy.misc.imsave(path, image)
    return mp.imsave(path, image)

def inverse_transform(images):
    return (images+1.)/2.

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def show_all_variables():
    model_vars = tf.trainable_variables()
    print("####Output all the trainable variables####")
    print("The count is: "+ str(len(model_vars)))
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

