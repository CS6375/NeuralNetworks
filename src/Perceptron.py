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
        self.__weight__ = [w + self.eta * self.__delta__ * x
                           for w, x in zip(self.__weight__, inputs_with_bias)]

    def compute_net(self, attributes: List[float]) -> float:
        return sum([a * b for a, b in zip(attributes + [1], self.__weight__)])

    @property
    def delta(self) -> float:
        return self.__delta__

    @property
    def eta(self) -> float:
        return self.__learning_rate__

    @property
    def weights(self) -> List[float]:
        return self.__weight__

    @property
    def weight_delta(self) -> List[float]:
        """
        Return the list of (weight * delta) for each weight except the first
        one, i.e., weight for bias (which is used internally).
        :return:
        """
        return [self.__delta__ * w for w in self.__weight__[1:]]

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
