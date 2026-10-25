"""
Un modelo de lenguaje diminuto, a nivel de caracter, con la forma de GPT (solo decodificador): aprende a predecir el caracter siguiente y,
generando de a uno, escribe texto.

El texto son frases de una gramatica pequena ("el gato come pez.", "el perro bebe agua."...). Hay algo que aprender: tras "el gato come "
solo puede venir "pez". Mientras entrena, cada cierto numero de pasos escribe una muestra, para ver como pasa de caracteres al azar a
frases con sentido, poco a poco.

  python -m transformer.char_lm [--steps 1500] [--sample-every 250] [--layers 2] [--d-model 24]
"""
import argparse
import math
import random
import time

from transformer.models import LanguageModel
from transformer.tasks import Vocabulary, corpus, sentences
from transformer.training import evaluateLM, trainLM


def lossFloor(text, context):
    """
    Perdida minima aproximada del protocolo de entrenamiento. En un trozo de `context` simbolos, la posicion i solo ve i + 1 simbolos de
    historia, asi que las primeras posiciones tienen una incertidumbre inevitable. Se estima con la entropia condicional empirica del
    texto dado un historial de h simbolos, para h = 1..context, promediada sobre las `context` posiciones.
    """
    import collections
    total_floor = 0.0
    for h in range(1, context + 1):
        counts = collections.defaultdict(collections.Counter)
        for i in range(h, len(text)):
            counts[text[i - h:i]][text[i]] += 1
        n_total, entropy = 0, 0.0
        for c in counts.values():
            n = sum(c.values())
            n_total += n
            entropy -= sum(k * math.log(k / n) for k in c.values())
        total_floor += entropy / n_total
    return total_floor / context


def validSentences(text):
    """(frases validas, frases completas) en un texto generado: una frase es valida si pertenece a la gramatica."""
    valid = set(sentences())
    parts = [p.strip() + '.' for p in text.split('.')[:-1] if p.strip()]
    return sum(p in valid for p in parts), len(parts)


def sample(model, vocab, n, prompt='el ', seed=0, temperature=0.8):
    rng = random.Random(seed)
    return vocab.decode(model.generate(vocab.encode(prompt), n, rng, temperature))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--d-model', type=int, default=24)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=48)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--context', type=int, default=20, help='cuantos caracteres ve el modelo para predecir el siguiente')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prompt', default='el ')
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    text = corpus()
    vocab = Vocabulary(text)
    tokens = vocab.encode(text)
    model = LanguageModel(len(vocab), args.d_model, args.heads, args.d_ff, args.layers, max_len=args.context, seed=args.seed)   # max_len = contexto: al generar solo se usan posiciones que se vieron al entrenar
    print(f'Texto: {len(text)} caracteres, {len(vocab)} simbolos distintos {"".join(vocab.chars)!r}.  Frases posibles: {len(sentences())}.')
    print(f'Modelo: {args.layers} bloque(s), d_model={args.d_model}, {args.heads} cabezas, d_ff={args.d_ff}, contexto de {args.context}: {model.paramCount()} parametros.')
    print(f'Perdida: al azar seria {math.log(len(vocab)):.2f}; el minimo posible con este contexto es ~{lossFloor(text, args.context):.2f} '
          f'(las primeras posiciones de cada trozo ven poca historia y no se pueden predecir bien).')
    print('Antes de entrenar:')
    print(f'  {sample(model, vocab, 60, args.prompt, args.seed, args.temperature)!r}\n')

    start, done = time.time(), 0
    chunk = args.sample_every
    while done < args.steps:
        n = min(chunk, args.steps - done)
        losses = trainLM(model, tokens, args.context, steps=n, lr=args.lr, seed=args.seed + done)
        done += n
        loss, acc = evaluateLM(model, tokens, args.context)
        text_out = sample(model, vocab, 80, args.prompt, args.seed, args.temperature)
        print(f'paso {done:>5} | perdida {loss:.3f} | siguiente caracter acertado {acc:.0%} | {time.time() - start:.0f} s')
        print(f'  {text_out!r}', flush=True)

    long_text = sample(model, vocab, 600, args.prompt, args.seed + 1, args.temperature)
    ok, total = validSentences(long_text)
    print(f'\nEn 600 caracteres generados: {ok} de {total} frases completas pertenecen a la gramatica ({ok / max(total, 1):.0%}).')
    print(f'Frases de la gramatica: {sentences()}')


if __name__ == '__main__':
    main()
