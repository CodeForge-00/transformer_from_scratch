"""
Experimento: el transformer sobre la tarea de invertir una secuencia, la misma que se uso en attention_from_scratch con GRU.

Se entrena un codificador-decodificador (autoatencion + atencion cruzada + redes densas + residuales + LayerNorm) y se mide la exactitud
en secuencias nuevas, al alargar la secuencia. Con --show se dibuja adonde mira la atencion cruzada del ultimo bloque del decodificador.

  python -m transformer.reverse_task [--lengths 4 8 12] [--steps 1000] [--seeds 3] [--d-model 16] [--show]
"""
import argparse
import time

from transformer.models import Seq2SeqTransformer
from transformer.training import evaluateSeq2Seq, trainSeq2Seq

VOCAB = 6


def exampleSource(length, seed=5):
    import random
    rng = random.Random(seed)
    return [rng.randrange(VOCAB) for _ in range(length)]


def heatmap(weights, src, out):
    """Mapa de calor en texto: fila = simbolo que se genera, columna = posicion de la entrada. Mas oscuro = mas peso."""
    shades = ' .:-=+*#%@'
    lines = ['       entrada: ' + ' '.join(f'{s:>2}' for s in src), '       posicion:' + ' '.join(f'{j + 1:>2}' for j in range(len(src)))]
    for t, (row, token) in enumerate(zip(weights, out)):
        cells = ' '.join(' ' + shades[min(len(shades) - 1, int(w * len(shades)))] for w in row)
        best = max(range(len(row)), key=row.__getitem__)
        lines.append(f'  genera {token} (paso {t + 1:>2}): {cells}   mira la posicion {best + 1}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=['reverse', 'copy'], default='reverse')
    parser.add_argument('--lengths', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--d-model', type=int, default=16)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=32)
    parser.add_argument('--layers', type=int, default=1, help='bloques en el codificador y en el decodificador')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--show', action='store_true', help='al final, el mapa de calor de la atencion cruzada de un ejemplo')
    args = parser.parse_args()
    probe = Seq2SeqTransformer(VOCAB, args.d_model, args.heads, args.d_ff, args.layers, args.layers)
    print(f'Tarea: {args.task}. Vocabulario de {VOCAB} simbolos. Transformer de {probe.paramCount()} parametros '
          f'(d_model={args.d_model}, {args.heads} cabezas, d_ff={args.d_ff}, {args.layers} bloque(s) por lado), {args.steps} pasos de Adam, {args.seeds} semilla(s).')
    print('Exactitud en 200 secuencias nuevas, por simbolo / por secuencia completa; entre corchetes, cada semilla por separado:\n')
    models = {}
    for T in args.lengths:
        start, results = time.time(), []
        for seed in range(args.seeds):
            model = Seq2SeqTransformer(VOCAB, args.d_model, args.heads, args.d_ff, args.layers, args.layers, max_len=max(args.lengths) + 2, seed=seed)
            trainSeq2Seq(model, args.task, T, VOCAB, steps=args.steps, lr=args.lr, seed=seed)
            results.append(evaluateSeq2Seq(model, args.task, T, VOCAB))
            models[T] = model
        tok = sum(r[0] for r in results) / len(results)
        seq = sum(r[1] for r in results) / len(results)
        per_seed = ' '.join(f'{r[0]:.0%}/{r[1]:.0%}' for r in results)
        print(f'T={T:<3} {tok:>5.0%} / {seq:>5.0%}   [{per_seed}]   ({time.time() - start:.0f} s)', flush=True)
    if args.show:
        for T, model in models.items():
            src = exampleSource(T)
            out, weights = model.greedy(src, T)
            for h, head in enumerate(weights):
                print(f'\nAtencion cruzada, T={T}, cabeza {h + 1}:\n' + heatmap(head, src, out))


if __name__ == '__main__':
    main()
