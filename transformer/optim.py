"""Los dos optimizadores, con sus formulas a la vista."""
import math

from transformer.mathlib import isMatrix, paramsLike


def rows(p):
    """Un vector es una sola fila; una matriz, sus filas. Asi se recorren los dos igual (las filas se modifican en su sitio)."""
    return p if isMatrix(p) else [p]


class SGD:
    """Descenso del gradiente simple:  theta -= lr * g."""

    def __init__(self, param_dicts, lr=0.5):
        self.dicts, self.lr = param_dicts, lr

    def step(self, grad_dicts):
        for params, grads in zip(self.dicts, grad_dicts):
            for name in params:
                for p_row, g_row in zip(rows(params[name]), rows(grads[name])):
                    for j in range(len(p_row)):
                        p_row[j] -= self.lr * g_row[j]


class Adam:
    """
    Adam (Kingma y Ba, 2014). Para cada parametro theta con gradiente g:
        m = b1 m + (1 - b1) g            (media movil del gradiente)
        v = b2 v + (1 - b2) g^2          (media movil del gradiente al cuadrado)
        theta -= lr * m_hat / (sqrt(v_hat) + eps)       con m_hat = m / (1 - b1^t), v_hat = v / (1 - b2^t)
    """

    def __init__(self, param_dicts, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.dicts, self.lr, self.b1, self.b2, self.eps, self.t = param_dicts, lr, b1, b2, eps, 0
        self.m = [paramsLike(p) for p in param_dicts]
        self.v = [paramsLike(p) for p in param_dicts]

    def step(self, grad_dicts):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for params, grads, m, v in zip(self.dicts, grad_dicts, self.m, self.v):
            for name in params:
                for p_row, g_row, m_row, v_row in zip(rows(params[name]), rows(grads[name]), rows(m[name]), rows(v[name])):
                    for j in range(len(p_row)):
                        g = g_row[j]
                        m_row[j] = self.b1 * m_row[j] + (1 - self.b1) * g
                        v_row[j] = self.b2 * v_row[j] + (1 - self.b2) * g * g
                        p_row[j] -= self.lr * (m_row[j] / c1) / (math.sqrt(v_row[j] / c2) + self.eps)
