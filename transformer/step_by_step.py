"""
Ejemplos numericos pequenos de lo que anade un transformer a la atencion.

  python -m transformer.step_by_step

  A. LayerNorm de un vector (y su retropropagacion)
  B. Codificacion de posiciones
  C. Un bloque entero: el "flujo residual" paso a paso
  D. La mascara causal: el futuro no afecta al pasado
"""
import math

from transformer.blocks import TransformerBlock, positionalEncoding
from transformer.mathlib import add, makeRng, mean
from transformer.module import LayerNorm
from transformer.models import LanguageModel


def fmt(v):
    return '[' + ', '.join(f'{x:7.4f}' for x in v) + ']'


def partA():
    print('A. LAYERNORM de x = [1, 2, 3, 4]')
    x = [1.0, 2.0, 3.0, 4.0]
    mu, var = mean(x), mean([(v - mean(x)) ** 2 for v in x])
    print(f'   media     mu  = (1+2+3+4)/4 = {mu}')
    print(f'   varianza  var = ((-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2)/4 = {var}      desviacion = sqrt(var) = {math.sqrt(var):.4f}')
    ln = LayerNorm(4)
    y, cache = ln.forward([x])
    print(f'   x_hat = (x - mu) / sqrt(var + eps) = {fmt(y[0])}      (media 0, varianza 1; con gamma = 1 y beta = 0, y = x_hat)')
    ln.zeroGrad()
    dx = ln.backward(cache, [[1.0, 0.0, 0.0, 0.0]])
    print('   Hacia atras, con dL/dy = [1, 0, 0, 0]:')
    print(f'     dL/dx = {fmt(dx[0])}      suma = {sum(dx[0]):.4f}')
    print('     (suman 0: sumar una constante a todo x no cambia x_hat, asi que ningun reparto de dL/dx puede "subir" todo a la vez)')


def partB():
    print('\nB. CODIFICACION DE POSICIONES sinusoidal, d = 4')
    print('   PE[pos][2i] = sin(pos / 10000^(2i/d)),  PE[pos][2i+1] = cos(pos / 10000^(2i/d))')
    for pos, row in enumerate(positionalEncoding(4, 4)):
        print(f'   posicion {pos}: {fmt(row)}')
    print('   Las dos primeras componentes giran rapido (periodo 2*pi) y las dos ultimas despacio (periodo 2*pi*100):')
    print('   juntas dan a cada posicion un "codigo" distinto, como las manecillas de un reloj.')


def partC():
    print('\nC. UN BLOQUE de transformer (pre-LN) sobre 3 vectores de d = 4, con pesos aleatorios')
    d = 4
    block = TransformerBlock(d, n_heads=2, d_ff=8, seed=3)
    rng = makeRng(1)
    xs = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(3)]
    n1, _ = block.ln1.forward(xs)
    a, _, _ = block.attn.forward(n1)
    x1 = [add(x, v) for x, v in zip(xs, a)]
    n2, _ = block.ln2.forward(x1)
    f, _ = block.ffn.forward(n2)
    ys, _ = block.forward(xs)
    print('   x_i                        entrada')
    print('   LN1(x_i)                   se normaliza antes de la atencion')
    print('   a_i = Atencion(LN1(x))     lo que la atencion propone sumar')
    print('   x1_i = x_i + a_i           CONEXION RESIDUAL: la entrada pasa intacta y se le suma una correccion')
    print('   f_i = FFN(LN2(x1))         lo que la red densa propone sumar')
    print('   y_i = x1_i + f_i           otra residual')
    for i in range(3):
        print(f'\n   posicion {i + 1}:')
        for name, v in (('x', xs[i]), ('LN1(x)', n1[i]), ('a', a[i]), ('x1 = x + a', x1[i]), ('f', f[i]), ('y = x1 + f', ys[i])):
            print(f'     {name:<11} {fmt(v)}')
    ok = max(abs(y - (x + av + fv)) for yi, xi, ai, fi in zip(ys, xs, a, f) for y, x, av, fv in zip(yi, xi, ai, fi))
    print(f'\n   Comprobacion: y = x + a + f  (diferencia maxima {ok:.1e}). Cada subcapa solo AGREGA una correccion al vector que ya habia.')


def partD():
    print('\nD. LA MASCARA CAUSAL: cambiar el futuro no cambia el pasado')
    rng = makeRng(2)
    model = LanguageModel(6, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=1)
    tokens = [rng.randrange(6) for _ in range(6)]
    changed = tokens[:4] + [(tokens[4] + 1) % 6, (tokens[5] + 2) % 6]
    a = model.logits(tokens)[0]
    b = model.logits(changed)[0]
    print(f'   texto A = {tokens}')
    print(f'   texto B = {changed}      (distinto solo en las posiciones 5 y 6)')
    for i in range(6):
        diff = max(abs(x - y) for x, y in zip(a[i], b[i]))
        print(f'   salida de la posicion {i + 1}: diferencia maxima entre A y B = {diff:.2e}' + ('   <- no depende del futuro' if i < 4 else '   <- cambia: ve el simbolo nuevo'))


def main():
    partA()
    partB()
    partC()
    partD()


if __name__ == '__main__':
    main()
