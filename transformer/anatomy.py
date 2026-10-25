"""
Anatomia de un transformer: de que piezas se compone y cuantos parametros tiene cada una, con la formula que cuadra el total.

  python -m transformer.anatomy [--d-model 16] [--heads 2] [--d-ff 32] [--enc 1] [--dec 1] [--vocab 6]
"""
import argparse

from transformer.models import LanguageModel, Seq2SeqTransformer


def blockParams(d, d_ff, cross):
    """Parametros de un bloque: atencion = 4 d^2 (Wq, Wk, Wv, Wo, sin sesgos); LayerNorm = 2 d; FFN = 2 d d_ff + d_ff + d."""
    attention = 4 * d * d
    norm = 2 * d
    ffn = 2 * d * d_ff + d_ff + d
    parts = {'autoatencion': attention, 'LayerNorm 1': norm}
    if cross:
        parts.update({'atencion cruzada': attention, 'LayerNorm cruzada': norm})
    parts.update({'LayerNorm 2': norm, 'FFN': ffn})
    return parts


def expectedSeq2Seq(vocab, d, d_ff, n_enc, n_dec):
    enc = sum(blockParams(d, d_ff, False).values()) * n_enc + 2 * d
    dec = sum(blockParams(d, d_ff, True).values()) * n_dec + 2 * d
    return vocab * d + (vocab + 1) * d + enc + dec + d * vocab + vocab


def expectedLM(vocab, d, d_ff, n_layers):
    return vocab * d + sum(blockParams(d, d_ff, False).values()) * n_layers + 2 * d + d * vocab + vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vocab', type=int, default=6)
    parser.add_argument('--d-model', type=int, default=16)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=32)
    parser.add_argument('--enc', type=int, default=1, help='bloques del codificador')
    parser.add_argument('--dec', type=int, default=1, help='bloques del decodificador')
    args = parser.parse_args()
    d, f, V = args.d_model, args.d_ff, args.vocab

    print(f'd_model = {d}, {args.heads} cabezas (d_head = {d // args.heads}), d_ff = {f}, vocabulario = {V}\n')
    print('UN BLOQUE (parametros):')
    for title, cross in (('codificador (o solo-decodificador)', False), ('decodificador con atencion cruzada', True)):
        parts = blockParams(d, f, cross)
        print(f'  {title}:  ' + ' + '.join(f'{k} {v}' for k, v in parts.items()) + f'  =  {sum(parts.values())}')
    print(f'\n  atencion = 4 d^2 = 4 x {d}^2 = {4 * d * d}      FFN = 2 d d_ff + d_ff + d = {2 * d * f + f + d}      LayerNorm = 2 d = {2 * d}')

    model = Seq2SeqTransformer(V, d, args.heads, f, args.enc, args.dec)
    print(f'\nMODELO CODIFICADOR-DECODIFICADOR ({args.enc} bloque(s) en el codificador, {args.dec} en el decodificador):')
    for name, module in (('embedding de la entrada', model.src_emb), ('embedding de la salida (con BOS)', model.tgt_emb),
                         ('codificador', model.encoder), ('decodificador', model.decoder), ('capa de salida', model.out)):
        print(f'  {name:<34} {module.paramCount():>7}')
    print(f'  {"TOTAL":<34} {model.paramCount():>7}     (formula: {expectedSeq2Seq(V, d, f, args.enc, args.dec)})')

    lm = LanguageModel(V, d, args.heads, f, args.enc)
    print(f'\nMODELO SOLO-DECODIFICADOR, estilo GPT ({args.enc} bloque(s)): {lm.paramCount()} parametros     (formula: {expectedLM(V, d, f, args.enc)})')

    print('\nCOSTE DE LA ATENCION: cada posicion mira a todas, asi que hay T x T puntuaciones por cabeza y por capa.')
    for T in (8, 64, 512, 4096):
        print(f'  secuencia de {T:>4} posiciones: {T * T:>10,} puntuaciones por cabeza'.replace(',', '.'))
    print('  (crece con el cuadrado de la longitud; por eso las secuencias largas son caras)')


if __name__ == '__main__':
    main()
