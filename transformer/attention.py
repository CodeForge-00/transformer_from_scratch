"""
Atencion de varias cabezas (la del proyecto attention_from_scratch, ahora como modulo con gradientes acumulados).

    Q_i = Wq xq_i      K_j = Wk xkv_j      V_j = Wv xkv_j
    cada cabeza h usa un trozo de d_head = d_model / n_heads componentes:
        s_ij = Q_i^h . K_j^h / sqrt(d_head)        w_i = softmax_j(s_i)        o_i^h = suma_j w_ij V_j^h
    c_i = [o_i^1 ; ... ; o_i^H]      y_i = Wo c_i

Con xkv = xq es AUTOATENCION; con xkv distinto, ATENCION CRUZADA. `mask[i][j]` (True = permitido) impide mirar posiciones.
"""
import math

from transformer.mathlib import (addInPlace, addOuter, dot, makeRng, matTVec, matVec, randomMatrix, scale, softmax, softmaxBackward, zeros)
from transformer.module import Module

NEG_INF = float('-inf')


def causalMask(n):
    """mask[i][j] = True si la posicion i puede mirar a la j: solo a las anteriores y a si misma (j <= i)."""
    return [[j <= i for j in range(n)] for i in range(n)]


def attend(scores, values, allowed=None):
    """pesos = softmax(puntuaciones) (con -inf donde no se puede mirar);  contexto = suma_i pesos_i * valor_i."""
    if allowed is not None:
        scores = [s if ok else NEG_INF for s, ok in zip(scores, allowed)]
    weights = softmax(scores)
    context = zeros(len(values[0]))
    for w, v in zip(weights, values):
        addInPlace(context, scale(v, w))
    return context, weights


def attendBackward(weights, values, dcontext):
    """dL/dv_i = w_i dL/dc ;  dL/dw_i = dL/dc . v_i ; despues, por el jacobiano del softmax, dL/dpuntuaciones."""
    dvalues = [scale(dcontext, w) for w in weights]
    dweights = [dot(dcontext, v) for v in values]
    return softmaxBackward(weights, dweights), dvalues


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads=1, seed=0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model debe ser multiplo de n_heads'
        rng, limit = makeRng(seed), 1 / math.sqrt(d_model)
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.params = {name: randomMatrix(d_model, d_model, limit, rng) for name in ('Wq', 'Wk', 'Wv', 'Wo')}
        self.initGrads()

    def forward(self, xq, xkv=None, mask=None):
        """Devuelve (salidas, pesos[cabeza][i][j], cache)."""
        p, H, dh = self.params, self.n_heads, self.d_head
        xkv = xq if xkv is None else xkv
        Q = [matVec(p['Wq'], x) for x in xq]
        K = [matVec(p['Wk'], x) for x in xkv]
        V = [matVec(p['Wv'], x) for x in xkv]
        factor = 1.0 / math.sqrt(dh)
        weights = [[None] * len(xq) for _ in range(H)]
        concat = [[] for _ in xq]
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            Kh, Vh = [k[sl] for k in K], [v[sl] for v in V]
            for i, q in enumerate(Q):
                out, w = attend([dot(q[sl], k) * factor for k in Kh], Vh, None if mask is None else mask[i])
                weights[h][i] = w
                concat[i] += out
        ys = [matVec(p['Wo'], c) for c in concat]
        return ys, weights, {'xq': xq, 'xkv': xkv, 'Q': Q, 'K': K, 'V': V, 'weights': weights, 'concat': concat}

    def backward(self, cache, dys):
        """Acumula los gradientes de los parametros y devuelve (dxq, dxkv). En autoatencion, la entrada recibe dxq + dxkv."""
        p, H, dh = self.params, self.n_heads, self.d_head
        Q, K, V, weights = cache['Q'], cache['K'], cache['V'], cache['weights']
        factor = 1.0 / math.sqrt(dh)
        dconcat = []
        for dy, c in zip(dys, cache['concat']):
            addOuter(self.grads['Wo'], dy, c)                    # y = Wo c  ->  dWo += dy c^T ;  dc = Wo^T dy
            dconcat.append(matTVec(p['Wo'], dy))
        dQ = [zeros(self.d_model) for _ in Q]
        dK = [zeros(self.d_model) for _ in K]
        dV = [zeros(self.d_model) for _ in V]
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            Kh, Vh = [k[sl] for k in K], [v[sl] for v in V]
            for i in range(len(Q)):
                dscores, dvals = attendBackward(weights[h][i], Vh, dconcat[i][sl])
                for j, (ds, dv) in enumerate(zip(dscores, dvals)):
                    ds *= factor
                    for t in range(dh):
                        dQ[i][h * dh + t] += ds * Kh[j][t]        # s_ij = Q_i . K_j  ->  dQ_i += ds K_j ,  dK_j += ds Q_i
                        dK[j][h * dh + t] += ds * Q[i][h * dh + t]
                        dV[j][h * dh + t] += dv[t]
        dxq = self._backProject('Wq', dQ, cache['xq'])
        dk = self._backProject('Wk', dK, cache['xkv'])
        dv = self._backProject('Wv', dV, cache['xkv'])
        return dxq, [[a + b for a, b in zip(u, w)] for u, w in zip(dk, dv)]

    def backwardSelf(self, cache, dys):
        dxq, dxkv = self.backward(cache, dys)
        return [[a + b for a, b in zip(u, w)] for u, w in zip(dxq, dxkv)]

    def _backProject(self, name, dZ, xs):
        """Z = W x  ->  dW += dZ x^T ;  dx = W^T dZ."""
        dxs = []
        for dz, x in zip(dZ, xs):
            addOuter(self.grads[name], dz, x)
            dxs.append(matTVec(self.params[name], dz))
        return dxs
