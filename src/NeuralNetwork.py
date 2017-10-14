import unittest
from functools import reduce
from typing import List, Callable

import Math
from Perceptron import Perceptron


class Layer:
    def __init__(self, dimension: int, previous_layer_dimension: int,
                 activation: Callable[[float], float]) -> None:
        self.__dimension__ = dimension
        self.__neurons__ = [Perceptron(dimension=previous_layer_dimension,
                                       activation=activation)
                            for _ in range(dimension)]
        self.__inputs__ = None

    def backward(self, downstream: List[float]):
        for p, d in zip(self.__neurons__, downstream):
            p.compute_delta(d)
        return reduce(lambda a, b: [x + y for x, y in zip(a, b)],
                      [n.weight_delta for n in self.__neurons__])

    def forward(self, inputs):
        self.__inputs__ = [1.] + inputs
        return [neuron.compute_output(self.__inputs__)
                for neuron in self.__neurons__]

    def update_weights(self):
        for neuron in self.__neurons__:
            neuron.update_weight(self.__inputs__)

    @property
    def inputs(self):
        return self.__inputs__


class NeuralNetwork:
    def __init__(self,
                 iteration: int,
                 layers: int,
                 input_count: int,
                 label_count: int,
                 neurons: List[int],
                 activation: Callable[[float], float] = Math.sigmoid) -> None:
        self.__iteration__ = iteration
        self.__layer_num__ = layers + 1
        self.__label_count__ = label_count
        neurons.append(label_count)
        self.__layers__ = [Layer(c, p, activation) for c, p in
                           zip(neurons, [input_count] + neurons)]

    def train_instance(self, instance):
        label = instance[-1]
        labels = [0 for _ in range(self.__label_count__)]
        labels[label] = 1

        inputs = instance[:-1]

        for layer in self.__layers__:
            inputs = layer.forward(inputs)

        deltas = [target - output for target, output in zip(labels, inputs)]

        for layer in self.__layers__[::-1]:
            deltas = layer.backward(deltas)

    def test_instance(self, instance) -> bool:
        label = instance[-1]

        inputs = instance[:-1]

        for layer in self.__layers__:
            inputs = layer.forward(inputs)

        return label == inputs.index(max(inputs))

    def __str__(self) -> str:
        ret = ''
        for i, layer in enumerate(self.__layers__):
            ret += 'Layer ' + str(i) + '\n'
            for j, n in enumerate(layer.__neurons__):
                ret += '\tNeuron ' + str(j) + ' weight: ' + str(n) + '\n'
        return ret


class TestNeuralNetwork(unittest.TestCase):
    """ This class is for testing the function of NeuralNetwork class. """

    def test_training_instance(self) -> None:
        nn = NeuralNetwork(200, 3, 3, 2, [3, 2, 1])

        nn.train_instance([1, 0, 1])

        print(nn)

    def test_test_instance(self) -> None:
        nn = NeuralNetwork(200, 3, 3, 2, [3, 2, 1])
        nn.train_instance([1, 0, 1])
        nn.test_instance([2, 1, 2])


if __name__ == '__main__':
    unittest.main()
