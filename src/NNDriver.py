#!/usr/bin/env python
"""Provides driver program for NeuralNetwork."""

import random
from sys import argv, stdout
from typing import List, Tuple

import pandas as pd

from NeuralNetwork import NeuralNetwork

__author__ = "Hanlin He (hxh160630), and Tao Wang (txw162630)"
__copyright__ = "Copyright 2017, The CS6375 Project"
__license__ = "Unlicense"
__version__ = "1.0.0"
__maintainer__ = "Hanlin He"
__email__ = "hxh160630@utdallas.edu"
__status__ = "Development"


def split(data_set: List, proportion: float) -> Tuple[List, List]:
    """
    First shuffle the ``data_set`` then split and return training set and
    testing set.
    :param data_set: DataSet to split.
    :param proportion: Proportion of training set.
    :return:
    """
    random.shuffle(data_set)
    length = len(data_set)
    training = int(proportion * length)
    return data_set[:training], data_set[training:]


def main():
    assert len(argv) >= 6

    data_file = argv[1]
    proportion = float(argv[2]) / 100
    iteration = int(argv[3])
    hidden_layer = int(argv[4])
    neurons = [int(argv[5 + i]) for i in range(hidden_layer)]

    pd_data_set = pd.read_csv(data_file)
    data_set = list(pd_data_set.as_matrix())

    input_count = len(data_set[0]) - 1
    label_count = len(set([instance[-1] for instance in data_set]))

    nn = NeuralNetwork(iteration=iteration,
                       input_count=input_count,
                       label_count=label_count,
                       hidden_layers=hidden_layer,
                       neurons=neurons)

    training_set, testing_set = split(data_set, proportion)

    # print('Training data set size: ', len(training_set))
    # print('Testing data set size: ', len(testing_set))
    stdout.write('Training data set size: %d \n' % len(training_set))
    stdout.flush()
    stdout.write('Testing data set size: %d \n\n' % len(testing_set))

    # actual_iteration, test_rate = nn.train(training_set)
    actual_iteration = 0
    test_rate = 0
    for _actual_iteration_, _test_rate_ in nn.train(training_set):
        # print(actual_iteration)
        actual_iteration = _actual_iteration_ + 1
        test_rate = _test_rate_
        stdout.write('Iteration Count: %d. Current training error: %.4f%%  \r' %
                     (actual_iteration, (1 - test_rate) * 100))
        stdout.flush()
    stdout.write('Total training iteration executed: %d\n' % actual_iteration)
    stdout.flush()
    stdout.write('Total training error = %.4f%% \n' % ((1 - test_rate) * 100))

    stdout.write('\nNeural Network model parameter are as follow:\n %s' %
                 str(nn))

    test_rate = nn.test(testing_set)

    stdout.write('\nTotal test error = %.4f%% \n' % ((1 - test_rate) * 100))


if __name__ == '__main__':
    main()
