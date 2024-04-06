import unittest
import numpy as np
from tensor import Tensor

class Ops(unittest.TestCase):
    def test_add(self):
        x = np.random.rand(5)
        y = np.random.rand(5)

        self.assertTrue(np.array_equal(x + y, (Tensor(x) + Tensor(y)).arr))

    def test_sub(self):
        x = np.random.rand(5)
        y = np.random.rand(5)

        self.assertTrue(np.array_equal(x - y, (Tensor(x) - Tensor(y)).arr))

    def test_mul(self):
        x = np.random.rand(5)
        y = np.random.rand(5)

        self.assertTrue(np.array_equal(x * y, (Tensor(x) * Tensor(y)).arr))
    
    def test_backprop(self):
        x = np.random.rand(5)
        y = np.random.rand(5)
        z = np.random.rand(5)

        x_t = Tensor(x)
        y_t = Tensor(y)
        z_t = Tensor(z)

        res = (x_t + y_t) * z_t
        random_grad = np.random.rand(5)
        res.grad = random_grad

        res.backward()

        self.assertTrue(np.array_equal(random_grad * z, x_t.grad))
        self.assertTrue(np.array_equal(random_grad * z, y_t.grad))
        self.assertTrue(np.array_equal(random_grad * (x + y), z_t.grad))

if __name__ == "__main__":
    unittest.main()
