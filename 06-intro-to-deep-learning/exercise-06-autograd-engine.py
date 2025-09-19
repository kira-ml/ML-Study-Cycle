import numpy as np
from collections import defaultdict

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
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _op='+', _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
            out._backward = _backward


        return out
    


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _op='*', _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
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
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _op='sum', _children=(self,))

        def _backward():
            if self.requires_grad:
                if axis is not None and not keepdims:
                    expanded_grad = np.expand_dims(out.grad, axis=axis)
                    self.grad += np.broadcast_to(expanded_grad, self.data.shape)
                else:
                    self.grad += np.broadcast_to(out.grad, self.data.shape)
        out._backward = _backward
        return out