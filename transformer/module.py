"""
Las piezas basicas de un transformer, cada una con `forward` y `backward` escritos a mano:

    Linear       y = W x + b                                       (una capa densa aplicada a cada posicion)
    LayerNorm    y = gamma (.) (x - media) / sqrt(varianza + eps) + beta     (normaliza CADA vector por separado)
    Embedding    tabla de vectores, uno por simbolo
    FeedForward  Linear -> ReLU -> Linear, igual para todas las posiciones

Convencion: una secuencia de n vectores es una lista de n listas. Cada pieza guarda sus parametros en `self.params` y acumula
sus gradientes en `self.grads` (con la misma estructura); `backward(cache, dys)` suma ahi y devuelve el gradiente de la entrada.
Asi, al juntar piezas, el optimizador solo tiene que recorrer `paramPairs()`.
"""
import math

from transformer.mathlib import (addInPlace, addOuter, dot, makeRng, matTVec, matVec, mean, paramCount, paramsLike, randomMatrix, relu,
                                 zeros, zeroGradsInPlace)


class Module:
    def __init__(self):
        self.params, self.grads, self.modules = {}, {}, []

    def initGrads(self):
        self.grads = paramsLike(self.params)

    def paramPairs(self):
        """[(parametros, gradientes), ...] de este modulo y de todos los que contiene."""
        pairs = [(self.params, self.grads)] if self.params else []
        for module in self.modules:
            pairs += module.paramPairs()
        return pairs

    def zeroGrad(self):
        for _, grads in self.paramPairs():
            zeroGradsInPlace(grads)

    def paramCount(self):
        return sum(paramCount(params) for params, _ in self.paramPairs())


class Linear(Module):
    """y = W x + b   con W de forma (salida x entrada)."""

    def __init__(self, n_in, n_out, seed=0, bias=True):
        super().__init__()
        rng = makeRng(seed)
        self.params = {'W': randomMatrix(n_out, n_in, 1 / math.sqrt(n_in), rng)}
        if bias:
            self.params['b'] = zeros(n_out)
        self.initGrads()

    def forward(self, xs):
        b = self.params.get('b')
        ys = [matVec(self.params['W'], x) for x in xs]
        if b is not None:
            ys = [[v + bi for v, bi in zip(y, b)] for y in ys]
        return ys, xs

    def backward(self, cache, dys):
        """y = W x + b  ->  dW += dy x^T,  db += dy,  dx = W^T dy."""
        dxs = []
        for x, dy in zip(cache, dys):
            addOuter(self.grads['W'], dy, x)
            if 'b' in self.grads:
                addInPlace(self.grads['b'], dy)
            dxs.append(matTVec(self.params['W'], dy))
        return dxs


class LayerNorm(Module):
    """
    Para cada vector x de d componentes:
        mu = media(x)      var = media((x - mu)^2)      x_hat = (x - mu) / sqrt(var + eps)      y = gamma (.) x_hat + beta
    Cada vector queda con media 0 y varianza 1 (antes de gamma y beta), sin depender del resto de la secuencia ni del lote.
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.params = {'gamma': [1.0] * d, 'beta': zeros(d)}
        self.initGrads()

    def forward(self, xs):
        ys, cache = [], []
        for x in xs:
            mu = mean(x)
            var = mean([(v - mu) ** 2 for v in x])
            inv = 1.0 / math.sqrt(var + self.eps)
            x_hat = [(v - mu) * inv for v in x]
            ys.append([g * xh + b for g, xh, b in zip(self.params['gamma'], x_hat, self.params['beta'])])
            cache.append((x_hat, inv))
        return ys, cache

    def backward(self, cache, dys):
        """
            dgamma += dy (.) x_hat        dbeta += dy        dx_hat = dy (.) gamma
            dx = (1 / sqrt(var + eps)) * ( dx_hat - media(dx_hat) - x_hat (.) media(dx_hat (.) x_hat) )
        Los dos terminos restantes aparecen porque mu y var tambien dependen de x.
        """
        dxs = []
        for (x_hat, inv), dy in zip(cache, dys):
            for j in range(len(dy)):
                self.grads['gamma'][j] += dy[j] * x_hat[j]
                self.grads['beta'][j] += dy[j]
            dx_hat = [d * g for d, g in zip(dy, self.params['gamma'])]
            m1 = mean(dx_hat)
            m2 = mean([d * xh for d, xh in zip(dx_hat, x_hat)])
            dxs.append([inv * (d - m1 - xh * m2) for d, xh in zip(dx_hat, x_hat)])
        return dxs


class Embedding(Module):
    """Tabla E (vocabulario x d): el simbolo t se convierte en la fila E[t]."""

    def __init__(self, vocab, d, seed=0):
        super().__init__()
        rng = makeRng(seed)
        self.params = {'E': [[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(vocab)]}
        self.initGrads()

    def forward(self, tokens):
        return [list(self.params['E'][t]) for t in tokens], tokens

    def backward(self, cache, dxs):
        """Solo las filas usadas reciben gradiente: dE[t] += dx."""
        for t, dx in zip(cache, dxs):
            addInPlace(self.grads['E'][t], dx)


class FeedForward(Module):
    """Red densa por posicion:  FFN(x) = W2 relu(W1 x + b1) + b2.  Se aplica igual y por separado a cada posicion."""

    def __init__(self, d_model, d_ff, seed=0):
        super().__init__()
        self.lin1, self.lin2 = Linear(d_model, d_ff, seed), Linear(d_ff, d_model, seed + 1)
        self.modules = [self.lin1, self.lin2]

    def forward(self, xs):
        hs, c1 = self.lin1.forward(xs)
        ys, c2 = self.lin2.forward([relu(h) for h in hs])
        return ys, (c1, c2, hs)

    def backward(self, cache, dys):
        c1, c2, hs = cache
        da = self.lin2.backward(c2, dys)
        dh = [[d if h > 0 else 0.0 for d, h in zip(row, hrow)] for row, hrow in zip(da, hs)]   # derivada de la ReLU: 1 si h > 0, si no 0
        return self.lin1.backward(c1, dh)
