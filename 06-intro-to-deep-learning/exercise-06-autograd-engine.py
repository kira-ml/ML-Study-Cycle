import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _op='', _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._op = _op
        self._children = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='+',
            _children=(self, other)
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='*',
            _children=(self, other)
        )

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='matmul',
            _children=(self, other)
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _op='relu', _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        sigmoid_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(sigmoid_data, requires_grad=self.requires_grad, _op='sigmoid', _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += sigmoid_data * (1 - sigmoid_data) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _op='sum',
                     _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def prune_tape(self):
        for child in self._children:
            if child.requires_grad:
                child.prune_tape()
        self._children = set()
        self._backward = lambda: None


class Linear:
    def __init__(self, input_size, output_size):
        self.weight = Tensor(np.random.randn(input_size, output_size) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros(output_size), requires_grad=True)

    def __call__(self, x):
        return x.matmul(self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]


def mse_loss(pred, target):
    return ((pred - target) ** 2).sum()


# Training example
np.random.seed(42)
X = Tensor(np.random.randn(100, 3))
y = Tensor(np.random.randn(100, 1))

model = Linear(3, 1)
learning_rate = 0.01

for epoch in range(100):
    pred = model(X)
    loss = mse_loss(pred, y)

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad
        param.grad.fill(0)

    loss.prune_tape()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Gradient check
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b + Tensor([1.0])
c.backward()
print(f"a: {a}, grad: {a.grad}")
print(f"b: {b}, grad: {b.grad}")
print(f"c: {c}")
