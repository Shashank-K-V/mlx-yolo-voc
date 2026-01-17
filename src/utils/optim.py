# src/utils/optim.py

import mlx.core as mx

class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, params, grads):
        return mx.tree_map(
            lambda p, g: p - self.lr * g,
            params,
            grads,
        )
