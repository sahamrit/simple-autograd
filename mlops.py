from typing import *

import numpy as np
class Add:

    @staticmethod
    def forward(x: np.array, y: np.array) -> np.array:
        res = x + y
        return res

    staticmethod
    def backward(grad: np.array, srcs: List[np.array]) -> None:
        return [grad, grad]

class Sub:

    @staticmethod
    def forward(x: np.array, y: np.array) -> np.array:
        res = x - y
        return res

    staticmethod
    def backward(grad: np.array, srcs: List[np.array]) -> None:
        return [grad, -grad]

class Mul:

    @staticmethod
    def forward(x: np.array, y: np.array) -> np.array:
        res = x * y
        return res

    staticmethod
    def backward(grad: np.array, srcs: List[np.array]) -> None:
        x, y = srcs
        return [grad * y, grad * x]

class Div:

    @staticmethod
    def forward(x: np.array, y: np.array) -> np.array:
        res = x / y
        return res

    staticmethod
    def backward(grad: np.array, srcs: List[np.array]) -> None:
        x, y = srcs
        return [grad / y, -grad * (x / (y * y))]