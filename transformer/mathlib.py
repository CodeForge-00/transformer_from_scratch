"""
Las operaciones matematicas que usa todo el proyecto, con listas de Python y el modulo `math` (sin NumPy ni otras bibliotecas).

Convencion: los vectores son columnas y las matrices tienen forma (filas, columnas); `W x` es el producto de la matriz W por el
vector x. Una secuencia de n vectores es una lista de n listas.
"""
import math
import random


# ---- vectores ----
def zeros(n):
    return [0.0] * n


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def hadamard(a, b):
    """Producto elemento a elemento: (a (.) b)_i = a_i * b_i."""
    return [x * y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def oneMinus(a):
    return [1.0 - x for x in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def argmax(a):
    return max(range(len(a)), key=a.__getitem__)


# ---- matrices ----
def zerosMatrix(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def randomMatrix(rows, cols, limit, rng):
    """Pesos uniformes en [-limit, limit]."""
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def matVec(M, v):
    """M v: el producto de cada fila de M con v."""
    return [dot(row, v) for row in M]


def matTVec(M, v):
    """M^T v: reparte un gradiente de la salida entre las entradas (se usa al retropropagar)."""
    result = [0.0] * len(M[0])
    for i, row in enumerate(M):
        vi = v[i]
        for j, m in enumerate(row):
            result[j] += m * vi
    return result


def addOuter(G, a, b):
    """G += a b^T, es decir G[i][j] += a[i] * b[j]. Asi se acumula el gradiente de una matriz de pesos."""
    for i, ai in enumerate(a):
        row = G[i]
        for j, bj in enumerate(b):
            row[j] += ai * bj


def addInPlace(a, b):
    """a += b (vectores)."""
    for i, v in enumerate(b):
        a[i] += v


# ---- activaciones ----
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoidVec(a):
    return [sigmoid(x) for x in a]


def tanhVec(a):
    return [math.tanh(x) for x in a]


# Las derivadas se escriben con la SALIDA y de la funcion, que el forward ya guardo:  sigmoide: y(1 - y)   tanh: 1 - y^2
def dSigmoid(y):
    return [v * (1.0 - v) for v in y]


def dTanh(y):
    return [1.0 - v * v for v in y]


# ---- softmax ----
def softmax(scores):
    """
    p_i = exp(s_i) / suma_j exp(s_j). Se resta el maximo antes de exponenciar (no cambia el resultado y evita desbordes).
    Una puntuacion de -inf da peso exactamente 0 (asi se enmascaran posiciones); al menos una debe ser finita.
    """
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def softmaxBackward(p, dp):
    """
    Dada la salida p = softmax(s) y dL/dp, devuelve dL/ds. El jacobiano del softmax es  dp_i/ds_j = p_i (delta_ij - p_j),
    asi que  dL/ds_j = p_j * ( dL/dp_j - suma_i p_i dL/dp_i ).
    """
    inner = sum(pi * dpi for pi, dpi in zip(p, dp))
    return [pj * (dpj - inner) for pj, dpj in zip(p, dp)]


# ---- parametros como diccionarios {nombre: vector o matriz} ----
def isMatrix(p):
    return len(p) > 0 and isinstance(p[0], list)


def paramsLike(params):
    """Diccionario con la misma estructura y todo a cero (para acumular gradientes)."""
    return {name: (zerosMatrix(len(p), len(p[0])) if isMatrix(p) else zeros(len(p))) for name, p in params.items()}


def paramCount(params):
    return sum(len(p) * len(p[0]) if isMatrix(p) else len(p) for p in params.values())


def paramNorm(*grad_dicts):
    """Norma global (raiz de la suma de cuadrados de todos los elementos): sirve para recortar el gradiente."""
    total = 0.0
    for grads in grad_dicts:
        for g in grads.values():
            rows = g if isMatrix(g) else [g]
            total += sum(v * v for row in rows for v in row)
    return math.sqrt(total)


def scaleGrads(grads, factor):
    for g in grads.values():
        for i, row in enumerate(g):
            if isinstance(row, list):
                g[i] = [v * factor for v in row]
            else:
                g[i] = row * factor


def makeRng(seed):
    return random.Random(seed)


def addGrads(total_list, grads_list, factor=1.0):
    """total += factor * grads, diccionario a diccionario (para promediar los gradientes de un lote)."""
    for total, grads in zip(total_list, grads_list):
        for name, g in grads.items():
            if isMatrix(g):
                for trow, grow in zip(total[name], g):
                    for j, v in enumerate(grow):
                        trow[j] += factor * v
            else:
                for j, v in enumerate(g):
                    total[name][j] += factor * v


# ---- extras del transformer ----
def mean(a):
    return sum(a) / len(a)


def relu(a):
    return [x if x > 0 else 0.0 for x in a]


def zeroGradsInPlace(grads):
    """Pone a cero todos los gradientes acumulados (sin crear listas nuevas, para que las referencias sigan siendo validas)."""
    for g in grads.values():
        for i, row in enumerate(g):
            if isinstance(row, list):
                for j in range(len(row)):
                    row[j] = 0.0
            else:
                g[i] = 0.0
