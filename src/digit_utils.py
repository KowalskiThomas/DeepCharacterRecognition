import h5py
import scipy.ndimage
import numpy

import os

import sys
import numpy

import pickle

import PIL
from PIL import Image

numpy.set_printoptions(threshold=sys.maxsize)

import random

import scipy.misc


def get_all_files(dir="numbers"):
    dir = os.path.abspath(dir)
    print(dir)

    all_files = list()
    for directory in os.listdir(dir):
        path = dir + "/" + directory
        if not os.path.isdir(path):
            continue

        for digit_dir in os.listdir(path):
            digit_path = path + "/" + digit_dir

            if not os.path.isdir(digit_path):
                continue

            for file in os.listdir(digit_path):
                if not "png" in file:
                    continue

                file_path = digit_path + "/" + file
                all_files.append((int(digit_dir), file_path))

    return all_files


def load_data(file_name: str):
    im = scipy.ndimage.imread(file_name, flatten=True)
    im = scipy.misc.imresize(im, (96, 96))
    return im


def apply_max(arr):
    out = []
    for line in arr:
        line = list(map(lambda x: 0 if x < 230 else 255, line))
        out.append(line)

    out = numpy.array(out)
    out.reshape(arr.shape)
    return out


def resize(arr):
    pass


def write_dataset():
    all_files = get_all_files("../numbers")
    n = len(all_files)

    x_data = numpy.zeros(shape=(len(all_files), 96, 96))
    y_data = numpy.zeros(shape=(len(all_files), 1))
    for i, (label, f) in enumerate(all_files):
        data = load_data(f)
        # scipy.misc.toimage(data).save("nonsat.png")
        data = apply_max(data)
        # scipy.misc.toimage(data).save("sat.png")
        y_data[i] = label
        x_data[i] = data

        if i % 100 == 0:
            print(i / n * 100, "%")

        # if i >= 100:
        #     break

    # file_names = numpy.array(list())
    # for file_name, x in all_data.items():
    #     y = all_labels[file_name]
    #     file_names.append(file_name)
    #     x_data.append(x_data)
    #     y_data.append(y_data)

    print("Saving compiled image data")
    with open("data.pckl", 'wb') as f:
        pickle.dump({
            "x": x_data,
            "y": y_data,
        }, f)
    print("OK")


def load_written_dataset():
    print("Loading: Opening file")
    with open("data.pckl", 'rb') as f:
        data = pickle.load(f)

    print("Loading: Shuffling")
    indices = [x for x in range(len(data["x"]))]
    random.shuffle(indices)

    # k = 983
    # scipy.misc.toimage(data["x"][k]).save("test.png")
    # print(data["y"][k])

    x = data["x"][indices]
    y = data["y"][indices]
    # x = [data["x"][i] for i in indices]
    # y = [data["y"][i] for i in indices]

    n = len(x)

    assert len(x) == len(y)

    dataset_divider = 2
    n = n // dataset_divider
    x = x[:n]
    y = y[:n]

    print(x.shape)
    print(y.shape)

    train_multiplier = 95
    train_divider = 100
    assert train_multiplier / train_divider < 1

    print("Loading: Converting")
    train_count = n // train_divider * train_multiplier
    print("Loading: Train count:", train_count)
    print("Loading: Test  count:", n - train_count)
    x_train = numpy.array(x[:train_count])
    x_test = numpy.array(x[train_count:])
    y_train = numpy.array(y[:train_count])
    y_test = numpy.array(y[train_count:])
    classes = numpy.array(range(11))

    return x_train, y_train, x_test, y_test, classes


if __name__ == '__main__':
    if "create" in sys.argv[1:]:
        print("Writing")
        write_dataset()
    else:
        load_written_dataset()
