#!/usr/bin/env python
"""Provides Perceptron, classes for linear classifier.

Perceptron is an algorithm for supervised learning of binary classifiers
(functions that can decide whether an input, represented by a vector of
numbers, belongs to some specific class or not). It is a type of linear
classifier, i.e. a classification algorithm that makes its predictions based
on a linear predictor function combining a set of weights with the feature
vector. The algorithm allows for online learning, in that it processes
elements in the training set one at a time.
"""

import random
import unittest
from typing import List, Callable

import Math

__author__ = "Hanlin He (hxh160630), and Tao Wang (txw162630)"
__copyright__ = "Copyright 2017, The CS6375 Project"
__license__ = "Unlicense"
__version__ = "1.0.0"
__maintainer__ = "Hanlin He"
__email__ = "hxh160630@utdallas.edu"
__status__ = "Development"


class Perceptron:
    def __init__(self,
                 dimension: int,
                 weights: List[float] = None,
                 learning_rate: float = 0.1,
                 activation: Callable[[float], float] = Math.sign,
                 ) -> None:
        """
        Initialize a Perceptron with specific dimension, learning rate and
        activation function.
        :param dimension: dimension of input data, excluding bias
        :param weights: specific weights of all input, including the weight
        for bias. If not specify, random generated weights are used.
        Note that, the weights should be generated randomly. This parameter
        is used for verifying, since specific weight values are easy to debug.
        :param learning_rate: learning rate of the Perceptron. Default 0.1
        :param activation: activation function, default sign().
        """
        self.__dimension__ = dimension + 1

        if weights is None:
            self.__weight__ = [random.uniform(-0.05, 0.05)
                               for _ in range(self.__dimension__)]
        else:
            # If weights are specified, it should include the initial weight
            # for bias.
            assert len(weights) == self.__dimension__
            self.__weight__ = weights

        self.__learning_rate__ = learning_rate

        self.__activation__ = activation
        self.__activation_partial__ = Math.partial(self.__activation__)

        self.__bias__ = 1.0
        self.__delta__ = 0.0
        self.__output__ = 0.0

    def train_instance(self, data_instance: List[float]) -> None:
        """
        Train the perceptron with one instance.
        :param data_instance: DataSet instance to be trained with.
        :return: None
        """
        assert len(data_instance) == self.__dimension__

        label = data_instance[-1]
        inputs = [1.] + data_instance[:-1]

        self.compute_output(inputs)
        self.compute_delta(label - self.__output__)
        self.update_weight(inputs)

    def compute_output(self, inputs) -> float:
        """
        Add bias to inputs, compute output through net and activation
        function, store and return the outputs.
        :param inputs: list of input of current Perceptron.
        :return: output computed.
        """

        # First add constant bias in front of inputs values.
        inputs_with_bias = [self.__bias__] + list(inputs)

        # Summing up the product of each weight and input, including bias.
        net = sum([a * b for a, b in zip(inputs_with_bias, self.__weight__)])

        # Apply the activation function and store the output values.
        self.__output__ = self.__activation__(net)

        # Return the output.
        return self.__output__

    def compute_delta(self, inputs) -> float:
        """
        Compute delta by multiply inputs with activation_partial(output).
        :param inputs: Difference made by current Perceptron.
        :return: Computed delta.
        """
        self.__delta__ = self.__activation_partial__(self.__output__) * inputs
        return self.__delta__

    def update_weight(self, inputs: List[float]) -> None:

        # First add constant bias in front of inputs values.
        inputs_with_bias = [self.__bias__] + list(inputs)

        # Number of inputs including the bias should be the same as weights
        # number.
        assert len(inputs_with_bias) == len(self.__weight__)

        # w = w + eta * delta * x
        self.__weight__ = [w + self.__learning_rate__ * self.__delta__ * x
                           for w, x in zip(self.__weight__, inputs_with_bias)]

    @property
    def weight_delta(self) -> List[float]:
        """
        Return the list of (weight * delta) for each weight except the first
        one, i.e., weight for bias (which is used internally).
        :return:
        """
        return [self.__delta__ * w for w in self.__weight__[1:]]

    def __str__(self) -> str:
        return str([float('%.4f' % w) for w in self.__weight__])


class TestPerceptron(unittest.TestCase):
    """ This class is for testing the function of Route class. """

    def test_init(self) -> None:
        p = Perceptron(10)
        print(p)

    def test_training_instance(self) -> None:
        p = Perceptron(2, [-2., 1., 2.], 0.5)

        p.train_instance([0.5, 1.5, 1])
        self.assertEqual(p.__weight__, [-2., 1., 2.])

        p.train_instance([-0.5, 0.5, -1])
        self.assertEqual(p.__weight__, [-2., 1., 2.])

        p.train_instance([0.5, 0.5, 1])
        self.assertEqual(p.__weight__, [-1., 1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
