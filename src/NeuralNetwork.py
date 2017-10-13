import unittest
from functools import reduce
from math import exp
from typing import List, Callable

from Perceptron import Perceptron


class Layer:
    def __init__(self, dimension: int, previous_layer_dimension: int) -> None:
        self.__dimension__ = dimension
        self.__neurons__ = [Perceptron(previous_layer_dimension) for _ in
                            range(dimension)]
        self.__inputs__ = None

    def backward(self, downstream: List[float]):
        for p, d in zip(self.__neurons__, downstream):
            p.compute_delta(d)
        return self.get_weight_deltas()

    def forward(self, inputs):
        self.__inputs__ = [1.] + inputs
        return [neuron.compute_output(self.__inputs__)
                for neuron in self.__neurons__]

    def get_weight_deltas(self):
        return reduce(lambda a, b: [x + y for x, y in zip(a, b)],
                      [n.weight_delta for n in self.__neurons__])

    def update_weights(self):
        for neuron in self.__neurons__:
            neuron.update_weight(self.__inputs__)

    @property
    def inputs(self):
        return self.__inputs__


class NeuralNetwork:
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))

    def __init__(self,
                 iteration: int,
                 layers: int,
                 input_count: int,
                 label_count: int,
                 neurons: List[int],
                 activation: Callable[[float], float] = sigmoid) -> None:
        self.__iteration__ = iteration
        self.__layer_num__ = layers + 1
        self.__label_count__ = label_count
        self.__layers__ = [Layer(c, p) for c, p in
                           zip(neurons, [input_count] + neurons)]
        self.__layers__ += [Layer(label_count, neurons[-1])]

    def train_instance(self, instance):
        label = instance[-1]
        labels = [0 for _ in range(self.__label_count__)]
        labels[label] = 1

        attributes = instance[:-1]

        for layer in self.__layers__:
            attributes = layer.forward(attributes)

        ss = [label - output for label, output in
              zip(labels, attributes)]

        for layer in self.__layers__[::-1]:
            ss = layer.backward(ss)


class TestNeuralNetwork(unittest.TestCase):
    """ This class is for testing the function of NeuralNetwork class. """

    def test_training_instance(self) -> None:
        nn = NeuralNetwork(200, 3, 2, 2, [3, 2, 1])

        nn.train_instance([1, 0, 1])


if __name__ == '__main__':
    unittest.main()
