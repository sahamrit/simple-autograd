from typing import *

import numpy as np

class Tensor:
    def __init__(self, arr: List[List]) -> None:
        self.arr: np.array = np.array(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype
        self.grad: np.array = None
        self.srcs: List[Tensor] = None
        self.op = None