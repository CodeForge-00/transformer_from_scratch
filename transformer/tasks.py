"""
Datos de los experimentos.

Tareas de secuencia-a-secuencia (las mismas del proyecto attention_from_scratch):
    reverse  invertir la secuencia   [3, 1, 4, 2] -> [2, 4, 1, 3]
    copy     copiarla                [3, 1, 4, 2] -> [3, 1, 4, 2]

Modelo de lenguaje: un texto pequeno con estructura, generado con una gramatica de tres animales con sus comidas y bebidas, de modo que
hay algo que aprender (despues de "el gato come " solo puede venir "pez") pero cabe en un modelo diminuto.
"""
import random


def makeExample(task, length, vocab, rng):
    src = [rng.randrange(vocab) for _ in range(length)]
    return src, (src[::-1] if task == 'reverse' else list(src))


def makeSet(task, length, vocab, n, rng):
    return [makeExample(task, length, vocab, rng) for _ in range(n)]


FOODS = {'gato': ('pez', 'leche'), 'perro': ('carne', 'agua'), 'loro': ('maiz', 'agua')}


def sentences():
    """Todas las frases posibles de la gramatica: 'el gato come pez.', 'el gato bebe leche.', ..."""
    return [f'el {animal} {verb} {obj}.' for animal, (food, drink) in FOODS.items() for verb, obj in (('come', food), ('bebe', drink))]


def corpus(n_sentences=120, seed=0):
    """Texto de frases elegidas al azar, separadas por espacios."""
    rng = random.Random(seed)
    options = sentences()
    return ' '.join(rng.choice(options) for _ in range(n_sentences)) + ' '


class Vocabulary:
    """Simbolo <-> entero para un texto, a nivel de caracter."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.index[c] for c in text]

    def decode(self, tokens):
        return ''.join(self.chars[t] for t in tokens)
