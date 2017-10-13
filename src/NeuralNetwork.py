from Perceptron import Perceptron
import typing
from functools import reduce


class OutputUnit(Perceptron):
    def __init__(self,
                 dimension: int,
                 weights: typing.List[float] = None,
                 learning_rate: float = 0.1,
                 activation: typing.Callable[[float], float] = None) -> None:
        super().__init__(dimension, weights, learning_rate, activation)

    def compute_delta(self, inputs):
        self.__delta__ = self.__output__ * (1 - self.__output__) * inputs
        pass

    def set_label(self, label):
        self.__label__ = label


class HiddenUnit(Perceptron):
    def __init__(self,
                 dimension: int,
                 weights: typing.List[float] = None,
                 learning_rate: float = 0.1,
                 activation: typing.Callable[[float], float] = None) -> None:
        super().__init__(dimension, weights, learning_rate, activation)

    def compute_delta(self, inputs):
        self.__delta__ = self.__output__ * (1 - self.__output__) * inputs
        pass


class Layer:
    __neurons__ = None

    def __init__(self,
                 dimension: int,
                 previous_dimension:int,
                 unit_type:typing.Type[Perceptron]) -> None:
        self.__dimension__ = dimension
        self.__neurons__ = [unit_type(previous_dimension) for _ in
                            range(dimension)]

    def backward(self, downstream: typing.List[float]):
        s = sum(downstream)
        for p in self.__neurons__:
            p.compute_delta(s)
        return self.get_weight_deltas()

    def set_label(self, labels):
        for neuron, label in zip(self.__neurons__, labels):
            neuron.set_label(label)

    def get_outputs(self):
        return [n.get_output() for n in self.__neurons__]

    def forward(self, input_values):
        data = [1.] + input_values
        for neuron in self.__neurons__:
            neuron.compute_output(data)
        return self.get_outputs()

    def get_weight_deltas(self):
        return reduce(lambda a, b: [x + y for x, y in zip(a, b)],
                      [n.get_weight_delta for n in self.__neurons__])


class OutputLayer:
    pass

class HiddenLayer:
    def __init__(self, dimension, previous_dimension) -> None:
        self.__dimension__ = dimension
        self.__neurons__ = [HiddenUnit(previous_dimension) for _ in
                            range(dimension)]

    def backward(self, downstream: typing.List[float]):
        s = sum(downstream)
        for p in self.__neurons__:
            p.compute_delta(s)
        return self.get_weight_deltas()

    def forward(self, input_values):
        self.__data__ = [1.] + input_values
        for neuron in self.__neurons__:
            neuron.compute_output(self.__data__)
        return self.get_outputs()

    def get_outputs(self):
        return [n.get_output() for n in self.__neurons__]

    def get_weight_deltas(self):
        return reduce(lambda a, b: [x + y for x, y in zip(a, b)],
                      [n.get_weight_delta for n in self.__neurons__])

    def update_weights(self):
        for neuron in self.__neurons__:
            neuron.update_weight(self.__data__)


class NeuralNetwork:
    def __init__(self,
                 iteration: int,
                 layers: int,
                 input_count: int,
                 label_count: int,
                 neurons: typing.List[int]) -> None:
        self.__iteration__ = iteration
        self.__layer_num__ = layers + 1
        self.__label_count__ = label_count
        self.__layers__ = [HiddenLayer(c, p) for c, p in
                           zip(neurons, [input_count] + neurons)]
        self.__layers__ += [OutputLayer(label_count, neurons[-1])]

        l = Layer(1,2, HiddenUnit)

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
