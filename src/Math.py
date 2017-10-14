from math import exp
from typing import Callable

__partial_map__ = dict()


def sign(x):
    return 1 if x >= 0 else -1


def sign_partial(x):
    return x


__partial_map__[sign] = sign_partial


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def sigmoid_partial(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)


__partial_map__[sigmoid] = sigmoid_partial


def partial(f: Callable[[float], float]) -> Callable[[float], float]:
    return __partial_map__[f]
