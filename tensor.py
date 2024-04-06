from typing import *

import numpy as np
import mlops

class Tensor:
    def __init__(self, arr: List[List]) -> None:
        self.arr: np.array = np.array(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype
        self.grad: np.array = None
        self.srcs: List[Tensor] = None
        self.op = None

    def __add__(self, y):
        res = Tensor(mlops.Add.forward(self.arr, y.arr))
        res.op = mlops.Add
        res.srcs = [self, y]
        return res

    def __sub__(self, y):
        res = Tensor(mlops.Sub.forward(self.arr, y.arr))
        res.op = mlops.Sub
        res.srcs = [self, y]
        return res

    def __mul__(self, y):
        res = Tensor(mlops.Mul.forward(self.arr, y.arr))
        res.op = mlops.Mul
        res.srcs = [self, y]
        return res

    def __truediv__(self, y):
        res = Tensor(mlops.Div.forward(self.arr, y.arr))
        res.op = mlops.Div
        res.srcs = [self, y]
        return res

    def backward(self):
        grad: np.array = self.grad
        op = self.op
        src_grad = op.backward(grad, [x.arr for x in self.srcs])
        for i, src in enumerate(self.srcs):
            src.grad = src_grad[i]
            if src.srcs is not None:
                src.backward()