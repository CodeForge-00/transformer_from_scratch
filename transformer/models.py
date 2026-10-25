"""
Los dos modelos que se construyen con los bloques:

  Seq2SeqTransformer   CODIFICADOR-DECODIFICADOR, el del articulo "Attention Is All You Need". El codificador lee toda la entrada con
                       autoatencion (sin mascara). El decodificador genera la salida simbolo a simbolo: autoatencion CAUSAL sobre lo que
                       lleva generado + atencion CRUZADA a la salida del codificador. Aqui hace lo mismo que el modelo de GRU del
                       proyecto attention_from_scratch (invertir secuencias), asi que se pueden comparar.

  LanguageModel        SOLO DECODIFICADOR, la forma de GPT: autoatencion causal y nada mas. Aprende a predecir el simbolo siguiente a partir
                       de los anteriores, y generando uno cada vez produce texto.

En los dos: entrada = embedding(simbolo) + codificacion de posicion;  pila de bloques;  Linear -> softmax sobre el vocabulario.
"""
import math

from transformer.blocks import Stack, positionalEncoding
from transformer.mathlib import add, argmax, softmax
from transformer.module import Embedding, Linear, Module


def crossEntropy(logits, targets):
    """Perdida media por posicion y su gradiente respecto a los logits:  dL/dlogits = (softmax - onehot) / n."""
    n, loss, dlogits = len(targets), 0.0, []
    for z, t in zip(logits, targets):
        p = softmax(z)
        loss -= math.log(max(p[t], 1e-12))
        d = list(p)
        d[t] -= 1.0
        dlogits.append([v / n for v in d])
    return loss / n, dlogits


class Seq2SeqTransformer(Module):
    def __init__(self, vocab, d_model=16, n_heads=2, d_ff=32, n_enc=1, n_dec=1, max_len=32, seed=0):
        super().__init__()
        self.vocab, self.bos = vocab, vocab
        self.src_emb, self.tgt_emb = Embedding(vocab, d_model, seed), Embedding(vocab + 1, d_model, seed + 1)   # un simbolo mas: BOS ("empieza")
        self.encoder = Stack(n_enc, d_model, n_heads, d_ff, causal=False, cross=False, seed=seed + 100)
        self.decoder = Stack(n_dec, d_model, n_heads, d_ff, causal=True, cross=True, seed=seed + 200)
        self.out = Linear(d_model, vocab, seed + 300)
        self.pe = positionalEncoding(max_len, d_model)
        self.modules = [self.src_emb, self.tgt_emb, self.encoder, self.decoder, self.out]

    def _embed(self, table, tokens):
        xs, cache = table.forward(tokens)
        return [add(x, self.pe[i]) for i, x in enumerate(xs)], cache      # la codificacion de posicion se SUMA y no se entrena

    def forward(self, src, tgt):
        """Teacher forcing: el decodificador recibe BOS y los simbolos correctos anteriores. Devuelve (perdida media, cache)."""
        xs, c_src = self._embed(self.src_emb, src)
        memory, c_enc = self.encoder.forward(xs)
        dec_tokens = [self.bos] + list(tgt[:-1])
        ys, c_tgt = self._embed(self.tgt_emb, dec_tokens)
        hs, c_dec = self.decoder.forward(ys, memory)
        logits, c_out = self.out.forward(hs)
        loss, dlogits = crossEntropy(logits, tgt)
        return loss, {'src': src, 'dec_tokens': dec_tokens, 'c_src': c_src, 'c_enc': c_enc, 'c_tgt': c_tgt, 'c_dec': c_dec,
                      'c_out': c_out, 'dlogits': dlogits}

    def backward(self, cache):
        """Acumula los gradientes en todos los modulos (llamar a zeroGrad() antes de empezar un lote)."""
        dhs = self.out.backward(cache['c_out'], cache['dlogits'])
        dys, dmemory = self.decoder.backward(cache['c_dec'], dhs)
        self.tgt_emb.backward(cache['c_tgt'], dys)                         # la posicion no tiene parametros: su gradiente se descarta
        dxs, _ = self.encoder.backward(cache['c_enc'], dmemory)
        self.src_emb.backward(cache['c_src'], dxs)

    def greedy(self, src, length):
        """
        Genera `length` simbolos, eligiendo siempre el mas probable y realimentando lo generado. Devuelve (simbolos, pesos de la
        atencion cruzada del ultimo bloque del decodificador: pesos[cabeza][paso][posicion de la entrada]).
        """
        xs, _ = self._embed(self.src_emb, src)
        memory, _ = self.encoder.forward(xs)
        tokens = [self.bos]
        for _ in range(length):
            ys, _ = self._embed(self.tgt_emb, tokens)
            hs, (caches, _) = self.decoder.forward(ys, memory)
            tokens.append(argmax(self.out.forward([hs[-1]])[0][0]))
        ys, _ = self._embed(self.tgt_emb, tokens[:-1])
        _, (caches, _) = self.decoder.forward(ys, memory)
        return tokens[1:], caches[-1]['cross_weights']


class LanguageModel(Module):
    """Modelo de lenguaje solo-decodificador: p(simbolo siguiente | simbolos anteriores)."""

    def __init__(self, vocab, d_model=24, n_heads=2, d_ff=48, n_layers=2, max_len=32, seed=0):
        super().__init__()
        self.vocab, self.max_len = vocab, max_len
        self.emb = Embedding(vocab, d_model, seed)
        self.stack = Stack(n_layers, d_model, n_heads, d_ff, causal=True, cross=False, seed=seed + 100)
        self.out = Linear(d_model, vocab, seed + 300)
        self.pe = positionalEncoding(max_len, d_model)
        self.modules = [self.emb, self.stack, self.out]

    def logits(self, tokens):
        xs, c_emb = self.emb.forward(tokens)
        xs = [add(x, self.pe[i]) for i, x in enumerate(xs)]
        hs, c_stack = self.stack.forward(xs)
        logits, c_out = self.out.forward(hs)
        return logits, (c_emb, c_stack, c_out)

    def forward(self, tokens, targets):
        """tokens y targets tienen la misma longitud; targets[i] es el simbolo que sigue a tokens[i]. Devuelve (perdida media, cache)."""
        logits, cache = self.logits(tokens)
        loss, dlogits = crossEntropy(logits, targets)
        return loss, cache + (dlogits,)

    def backward(self, cache):
        c_emb, c_stack, c_out, dlogits = cache
        dhs = self.out.backward(c_out, dlogits)
        dxs, _ = self.stack.backward(c_stack, dhs)
        self.emb.backward(c_emb, dxs)

    def generate(self, prefix, n, rng, temperature=1.0):
        """Genera n simbolos a partir de un prefijo, muestreando de softmax(logits / temperatura) y realimentando cada uno."""
        tokens = list(prefix)
        for _ in range(n):
            window = tokens[-self.max_len:]
            z = self.logits(window)[0][-1]
            p = softmax([v / temperature for v in z])
            r, acc, choice = rng.random(), 0.0, len(p) - 1
            for i, pi in enumerate(p):
                acc += pi
                if r < acc:
                    choice = i
                    break
            tokens.append(choice)
        return tokens
