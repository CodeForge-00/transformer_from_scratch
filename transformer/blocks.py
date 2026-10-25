"""
El bloque del transformer y la codificacion de posiciones.

BLOQUE (variante "pre-LN", la de GPT-2: se normaliza ANTES de cada subcapa):

    x1 = x  + Atencion( LN1(x) )                     autoatencion (con mascara causal en un decodificador)
    x2 = x1 + AtencionCruzada( LN2(x1), memoria )    solo en el decodificador de un modelo codificador-decodificador
    y  = x2 + FFN( LN3(x2) )

Cada `x + subcapa(...)` es una CONEXION RESIDUAL: el vector original pasa intacto y la subcapa solo le suma una correccion. Su
derivada es 1 + d(subcapa)/dx, asi que el gradiente tiene un camino directo hasta la entrada (el mismo camino aditivo de la LSTM).
"""
import math

from transformer.attention import MultiHeadAttention, causalMask
from transformer.mathlib import add
from transformer.module import FeedForward, LayerNorm, Module


def positionalEncoding(n_positions, d_model):
    """PE[pos][2i] = sin(pos / 10000^(2i/d)),  PE[pos][2i+1] = cos(pos / 10000^(2i/d)): un patron distinto para cada posicion."""
    pe = []
    for pos in range(n_positions):
        row = []
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            row += [math.sin(angle), math.cos(angle)]
        pe.append(row[:d_model])
    return pe


class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff, causal=False, cross=False, seed=0):
        super().__init__()
        self.causal, self.cross = causal, cross
        self.ln1, self.attn = LayerNorm(d_model), MultiHeadAttention(d_model, n_heads, seed)
        self.modules = [self.ln1, self.attn]
        if cross:
            self.ln_cross, self.cross_attn = LayerNorm(d_model), MultiHeadAttention(d_model, n_heads, seed + 1)
            self.modules += [self.ln_cross, self.cross_attn]
        self.ln2, self.ffn = LayerNorm(d_model), FeedForward(d_model, d_ff, seed + 2)
        self.modules += [self.ln2, self.ffn]

    def forward(self, xs, memory=None):
        """Devuelve (ys, cache). La cache guarda tambien los pesos de atencion (para dibujarlos)."""
        mask = causalMask(len(xs)) if self.causal else None
        n1, c_ln1 = self.ln1.forward(xs)
        a, w_self, c_attn = self.attn.forward(n1, None, mask)
        x1 = [add(x, v) for x, v in zip(xs, a)]                          # conexion residual
        cache = {'ln1': c_ln1, 'attn': c_attn, 'self_weights': w_self, 'cross_weights': None}
        if self.cross:
            n2, c_lnc = self.ln_cross.forward(x1)
            c, w_cross, c_cross = self.cross_attn.forward(n2, memory)
            x2 = [add(x, v) for x, v in zip(x1, c)]
            cache.update({'ln_cross': c_lnc, 'cross': c_cross, 'cross_weights': w_cross})
        else:
            x2 = x1
        n3, c_ln2 = self.ln2.forward(x2)
        f, c_ffn = self.ffn.forward(n3)
        ys = [add(x, v) for x, v in zip(x2, f)]
        cache.update({'ln2': c_ln2, 'ffn': c_ffn})
        return ys, cache

    def backward(self, cache, dys):
        """Devuelve (dxs, dmemoria o None). En cada residual, el gradiente pasa directo (la parte `1`) y ademas por la subcapa."""
        d3 = self.ffn.backward(cache['ffn'], dys)
        dx2 = [add(dy, d) for dy, d in zip(dys, self.ln2.backward(cache['ln2'], d3))]
        dmemory = None
        if self.cross:
            dxq, dmemory = self.cross_attn.backward(cache['cross'], dx2)
            dx1 = [add(d, e) for d, e in zip(dx2, self.ln_cross.backward(cache['ln_cross'], dxq))]
        else:
            dx1 = dx2
        d_attn = self.attn.backwardSelf(cache['attn'], dx1)
        dx0 = [add(d, e) for d, e in zip(dx1, self.ln1.backward(cache['ln1'], d_attn))]
        return dx0, dmemory


class Stack(Module):
    """Varios bloques seguidos y una LayerNorm final (en pre-LN hace falta, porque la salida de la ultima suma no esta normalizada)."""

    def __init__(self, n_blocks, d_model, n_heads, d_ff, causal=False, cross=False, seed=0):
        super().__init__()
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, causal, cross, seed + 10 * i) for i in range(n_blocks)]
        self.final_ln = LayerNorm(d_model)
        self.modules = self.blocks + [self.final_ln]

    def forward(self, xs, memory=None):
        caches = []
        for block in self.blocks:
            xs, cache = block.forward(xs, memory)
            caches.append(cache)
        ys, c_final = self.final_ln.forward(xs)
        return ys, (caches, c_final)

    def backward(self, cache, dys):
        caches, c_final = cache
        dxs = self.final_ln.backward(c_final, dys)
        dmemory = None
        for block, c in zip(reversed(self.blocks), reversed(caches)):
            dxs, dm = block.backward(c, dxs)
            if dm is not None:
                dmemory = dm if dmemory is None else [add(a, b) for a, b in zip(dmemory, dm)]
        return dxs, dmemory
