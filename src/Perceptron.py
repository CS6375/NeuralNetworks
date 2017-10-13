import random
import typing
import unittest


class Perceptron:
    @staticmethod
    def __default_activation__(x):
        return 1 if x >= 0 else -1

    def __init__(self,
                 dimension: int,
                 weights: typing.List[float] = None,
                 learning_rate: float = 0.1,
                 activation: typing.Callable[[float], float] = None) -> None:
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
            assert len(weights) == dimension
            self.__weight__ = weights

        self.__eta__ = learning_rate

        if activation is None:
            self.__activation__ = self.__default_activation__
        else:
            self.__activation__ = activation

        self.__delta__ = None
        self.__output__ = None
        self.__label__ = None

    def training_rule(self, w, eta, delta, x):
        return w + eta * delta * x

    def train_instance(self, data_instance: typing.List[float]) -> None:
        assert len(data_instance) == self.__dimension__

        self.__label__ = data_instance[-1]
        data = [1.] + data_instance[:-1]

        self.compute_output(data)

        self.compute_delta()
        self.update_weight(data)

    def compute_output(self, data):
        net = self.compute_net(data)
        self.__output__ = self.__activation__(net)

    def compute_delta(self, inputs):
        self.__delta__ = self.__label__ - self.__output__

    def update_weight(self, data) -> None:
        self.__weight__ = [(lambda w, e, d, x: w + e * d * x)(w,
                                                              self.__eta__,
                                                              self.__delta__,
                                                              x)
                           for w, x in zip(self.__weight__, data)]

    def compute_net(self, attributes: typing.List[float]) -> float:
        return sum([a * b for a, b in zip(attributes + [1], self.__weight__)])

    @property
    def get_weights(self) -> typing.List[float]:
        return self.__weight__

    @property
    def get_weight_delta(self) -> typing.List[float]:
        return [self.__delta__ * w for w in self.__weight__]

    def get_output(self) -> float:
        return self.__output__

    def get_delta(self):
        return self.__delta__

    def __str__(self) -> str:
        return self.__weight__.__str__()


class TestPerceptron(unittest.TestCase):
    """ This class is for testing the function of Route class. """

    def test_init(self) -> None:
        p = Perceptron(10)
        print(p)

    def test_training_instance(self) -> None:
        p = Perceptron(3, [-2., 1., 2.], 0.5)

        p.train_instance([0.5, 1.5, 1])
        self.assertEqual(p.get_weights, [-2., 1., 2.])

        p.train_instance([-0.5, 0.5, -1])
        self.assertEqual(p.get_weights, [-2., 1., 2.])

        p.train_instance([0.5, 0.5, 1])
        self.assertEqual(p.get_weights, [-1., 1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
