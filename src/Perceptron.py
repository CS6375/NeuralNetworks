import random
import unittest
from typing import List, Callable

import Math


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
        :param dimension: dimension of data, excluding bias
        :param weights: specific weights of all input. If not specify,
        random generated weights are used.
        :param learning_rate: learning rate of the Perceptron. Default 0.1
        :param activation: activation function, default sign().
        """
        self.__dimension__ = dimension + 1

        if weights is None:
            self.__weight__ = [random.uniform(0, 1) for _ in range(dimension)]
        else:
            assert len(weights) == self.__dimension__
            self.__weight__ = weights

        self.__eta__ = learning_rate

        self.__activation__ = activation
        self.__activation_partial__ = Math.partial(self.__activation__)

        self.__delta__ = 0.0
        self.__output__ = 0.0

    def train_instance(self, data_instance: List[float]) -> None:
        assert len(data_instance) == self.__dimension__

        label = data_instance[-1]
        inputs = [1.] + data_instance[:-1]

        self.compute_output(inputs)
        self.compute_delta(label - self.__output__)
        self.update_weight(inputs)

    def compute_output(self, inputs) -> float:
        net = sum([a * b for a, b in zip(inputs, self.__weight__)])
        self.__output__ = self.__activation__(net)
        return self.__output__

    def compute_delta(self, inputs) -> float:
        self.__delta__ = self.__activation_partial__(inputs)
        return self.__delta__

    def update_weight(self, inputs) -> None:
        self.__weight__ = [w + self.eta * self.delta * x
                           for w, x in zip(self.__weight__, inputs)]

    def compute_net(self, attributes: List[float]) -> float:
        return sum([a * b for a, b in zip(attributes + [1], self.__weight__)])

    @property
    def delta(self) -> float:
        return self.__delta__

    @property
    def eta(self) -> float:
        return self.__eta__

    @property
    def weights(self) -> List[float]:
        return self.__weight__

    @property
    def weight_delta(self) -> List[float]:
        return [self.__delta__ * w for w in self.__weight__]

    @property
    def output(self) -> float:
        return self.__output__

    def __str__(self) -> str:
        return self.__weight__.__str__()


class TestPerceptron(unittest.TestCase):
    """ This class is for testing the function of Route class. """

    def test_init(self) -> None:
        p = Perceptron(10)
        print(p)

    def test_training_instance(self) -> None:
        p = Perceptron(2, [-2., 1., 2.], 0.5)

        p.train_instance([0.5, 1.5, 1])
        self.assertEqual(p.weights, [-2., 1., 2.])

        p.train_instance([-0.5, 0.5, -1])
        self.assertEqual(p.weights, [-2., 1., 2.])

        p.train_instance([0.5, 0.5, 1])
        self.assertEqual(p.weights, [-1., 1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
