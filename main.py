from tensor import Tensor
import mlops


x = Tensor(list(range(5)))
y = Tensor([4, 5, 6, 7, 8])

res = mlops.Add.forward(mlops.Add, x, y)

gt = [[0] * 5]
grad = 2 * res.arr

res.op.backward(grad, [x, y])

print(x.grad)