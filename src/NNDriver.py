import random
from sys import argv
from typing import List, Tuple

import pandas as pd

from NeuralNetwork import NeuralNetwork


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

    print('Training data set size: ', len(training_set))
    print('Testing data set size: ', len(testing_set))

    actual_iteration, test_rate = nn.train(training_set)
    print('Total training iteration executed:', actual_iteration)
    print('Total training error = ', test_rate)

    print("Neural Network model parameter are as follow: ", nn)

    test_rate = nn.test(testing_set)

    print("Total test error = ", test_rate)


if __name__ == '__main__':
    main()
