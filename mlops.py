from typing import *
from tensor import Tensor

import numpy as np
class Add:

    @staticmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = x.arr + y.arr
        out = Tensor(res)
        out.srcs = [x, y]
        out.op = self
        return out

    staticmethod
    def backward(grad: Tensor, srcs: List[Tensor]) -> None:
        x, y = srcs
        x.grad = grad
        y.grad = grad

class Sub:

    @staticmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = x.arr - y.arr
        out = Tensor(res)
        out.srcs = [x, y]
        out.op = self
        return out

    staticmethod
    def backward(grad: Tensor, srcs: List[Tensor]) -> None:
        x, y = srcs
        x.grad = grad
        y.grad = -grad

class Mul:

    @staticmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = np.multiply(x.arr, y.arr)
        out = Tensor(res)
        out.srcs = [x, y]
        out.op = self
        return out

    staticmethod
    def backward(grad: Tensor, srcs: List[Tensor]) -> None:
        x, y = srcs
        x.grad = grad * y.arr
        y.grad = x.arr * grad