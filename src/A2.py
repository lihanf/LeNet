#!/usr/bin/env python
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import time
import argparse

from utils import config
from tqdm import tqdm
import time
import struct
import math

from model.LeNet5 import LeNet5

# read the images and labels
def readDataset(dataset):
    (image, label) = dataset
    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (img, lbl)


# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,), (pad,), (pad,), (0,)), 'constant', constant_values=(0, 0))
    return X_pad

def normalize(image):
    image -= image.min()
    image = image / image.max()
    image = (image-np.mean(image))/np.std(image)
    return image

# initialization of the weights & bias
def initialize(kernel_shape):
    b_shape = (1, 1, 1, kernel_shape[-1]) if len(kernel_shape) == 4 else (kernel_shape[-1],)
    mu, sigma = 0, 0.1
    weight = np.random.normal(mu, sigma, kernel_shape)
    bias = np.ones(b_shape) * 0.01
    return weight, bias

# return random-shuffled mini-batches
def random_mini_batches(image, label, mini_batch_size=256, one_batch=False):
    m = image.shape[0]  # number of training examples
    mini_batches = []

    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation, :, :, :]
    shuffled_label = label[permutation]

    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)

    # Partition (shuffled_image, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)

    return mini_batches

def get_parser():
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def load_dataset(test_image_path, test_label_path, train_image_path, train_label_path):
    trainset = (train_image_path, train_label_path)
    testset = (test_image_path, test_label_path)

    # read data
    (train_image, train_label) = readDataset(trainset)
    (test_image, test_label) = readDataset(testset)

    # data preprocessing
    train_image_normalized_pad = normalize(zero_pad(train_image[:, :, :, np.newaxis], 2))
    test_image_normalized_pad = normalize(zero_pad(test_image[:, :, :, np.newaxis], 2))

    return (train_image_normalized_pad, train_label), (test_image_normalized_pad, test_label)


def train(model, train_data, test_data, num_epoch, lr_global_list, batch_size):
    # Training loops
    n_train = train_data[0].shape[0]
    n_test = test_data[0].shape[0]
    st = time.time()
    cost_last, count = np.Inf, 0
    err_rate_list = []
    for epoch in range(0, num_epoch):
        print("---------- epoch", epoch + 1, "begin ----------")
        lr_global = lr_global_list[epoch]
        # print info
        print("learning rate: {}".format(lr_global))
        print("batch size: {}".format(batch_size))
        print('Training: ')
        # loop over each batch
        ste = time.time()
        cost = 0
        mini_batches = random_mini_batches(train_data[0], train_data[1], batch_size)

        for i in tqdm(range(len(mini_batches))):
            batch_image, batch_label = mini_batches[i]
            loss = model.Forward_Propagation(batch_image, batch_label, 'train')
            cost += loss
            model.Back_Propagation(lr_global)
        print("Done, total cost of epoch {}: {}".format(epoch + 1, cost))

        error01_train, _ = model.Forward_Propagation(train_data[0], train_data[1], 'test')
        error01_test, _ = model.Forward_Propagation(test_data[0], test_data[1], 'test')

        err_rate_list.append([error01_train/n_train, error01_test/n_test])
        print("0/1 error of training set: ", error01_train, "/", n_train)
        print("0/1 error of testing set: ", error01_test, "/", n_test)
        print("Time used: ", time.time() - ste, "sec")
        print("---------- epoch", epoch + 1, "end ------------")
        with open('model_data_' + str(epoch) + '.pkl', 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    err_rate_list = np.array(err_rate_list).T
    print("Total time used: ", time.time() - st, "sec")

    return err_rate_list


def test(model_path, test_data):
    # read model
    with open(model_path, 'rb') as input_:
        model = pickle.load(input_)

    error01, class_pred = model.Forward_Propagation(test_data[0], test_data[1], 'test')
    print("error rate:", error01 / len(class_pred))


def main():
    global args
        # args = get_parser()
    args = {
        "batch_size": 256,
        "epoch": 20,
        "model_path": "model_data_19.pkl",
        "test_image_path": "dataset/MNIST/t10k-images-idx3-ubyte",
        "test_label_path": "dataset/MNIST/t10k-labels-idx1-ubyte",
        "train_image_path": "dataset/MNIST/train-images-idx3-ubyte",
        "train_label_path": "dataset/MNIST/train-labels-idx1-ubyte",
    }
    train_data, test_data = load_dataset(args["test_image_path"], args["test_label_path"], args["train_image_path"],
                                         args["train_label_path"])
    train_data = list(train_data)
    test_data = list(test_data)

    n_train = 60000
    n_test = 10000
    train_data[0] = train_data[0].reshape(60000,1,32,32)[:n_train]
    train_data[1] = train_data[1][:n_train]
    test_data[0] = test_data[0].reshape(10000,1,32,32)[:n_test]
    test_data[1] = test_data[1][:n_test]

    model = LeNet5()
    # set lr for each epoch
    lr_global_list = np.array([5e-4] * 5 + [2e-4] * 5 + [1e-4] * 10)
    err_rate_list = train(model, train_data, test_data, args["epoch"], lr_global_list, args["batch_size"])

    # This shows the error rate of training and testing data after each epoch
    x = np.arange(args["epoch"])
    plt.xlabel('epoches')
    plt.ylabel('error rate')
    plt.plot(x, err_rate_list[0])
    plt.plot(x, err_rate_list[1])
    plt.legend(['training data', 'testing data'], loc='upper right')
    plt.grid(True)
    plt.show()

    # test model
    test(args["model_path"], test_data)
if __name__ == '__main__':
    main()
