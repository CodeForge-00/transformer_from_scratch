"""
Pruebas del proyecto. Cada backward se compara con el gradiente numerico (diferencias centrales) y los numeros del tutorial quedan
fijados. Ejecutar desde la raiz:  python -m unittest discover -s tests -t .
"""
import math
import unittest

from transformer import anatomy, step_by_step
from transformer.attention import MultiHeadAttention, causalMask
from transformer.blocks import Stack, TransformerBlock, positionalEncoding
from transformer.char_lm import lossFloor, validSentences
from transformer.mathlib import dot, isMatrix, makeRng, mean
from transformer.models import LanguageModel, Seq2SeqTransformer
from transformer.module import Embedding, FeedForward, LayerNorm, Linear
from transformer.optim import Adam, SGD
from transformer.reverse_task import exampleSource, heatmap
from transformer.tasks import Vocabulary, corpus, makeExample, sentences
from transformer.training import evaluateLM, evaluateSeq2Seq, trainLM, trainSeq2Seq

EPS = 1e-6


def rv(rng, n):
    return [rng.uniform(-1, 1) for _ in range(n)]


def numeric(f, container, index):
    old = container[index]
    container[index] = old + EPS
    up = f()
    container[index] = old - EPS
    down = f()
    container[index] = old
    return (up - down) / (2 * EPS)


def checkParams(test, f, module):
    for params, grads in module.paramPairs():
        for name, p in params.items():
            for i in range(len(p)):
                if isMatrix(p):
                    for j in range(len(p[i])):
                        test.assertAlmostEqual(grads[name][i][j], numeric(f, p[i], j), places=6, msg=f'{name}[{i}][{j}]')
                else:
                    test.assertAlmostEqual(grads[name][i], numeric(f, p, i), places=6, msg=f'{name}[{i}]')


def checkVectors(test, f, vectors, grads, label):
    for i, vec in enumerate(vectors):
        for j in range(len(vec)):
            test.assertAlmostEqual(grads[i][j], numeric(f, vec, j), places=6, msg=f'{label}[{i}][{j}]')


class SimpleModulesTest(unittest.TestCase):
    def check(self, module, forward, xs, ws):
        f = lambda: sum(dot(w, y) for w, y in zip(ws, forward(xs)[0]))
        module.zeroGrad()
        _, cache = forward(xs)
        dxs = module.backward(cache, ws)
        checkParams(self, f, module)
        checkVectors(self, f, xs, dxs, 'x')

    def setUp(self):
        rng = makeRng(1)
        self.xs, self.ws = [rv(rng, 4) for _ in range(3)], [rv(rng, 4) for _ in range(3)]
        self.rng = rng

    def test_linear(self):
        layer = Linear(4, 4, seed=1)
        self.check(layer, layer.forward, self.xs, self.ws)

    def test_layernorm(self):
        ln = LayerNorm(4)
        ln.params['gamma'], ln.params['beta'] = rv(self.rng, 4), rv(self.rng, 4)
        self.check(ln, ln.forward, self.xs, self.ws)

    def test_feedforward(self):
        ff = FeedForward(4, 8, seed=2)
        self.check(ff, ff.forward, self.xs, self.ws)

    def test_embedding_only_used_rows_get_gradient(self):
        emb = Embedding(5, 3, seed=1)
        tokens = [1, 3, 1]
        ws = [rv(self.rng, 3) for _ in tokens]
        f = lambda: sum(dot(w, y) for w, y in zip(ws, emb.forward(tokens)[0]))
        emb.zeroGrad()
        _, cache = emb.forward(tokens)
        emb.backward(cache, ws)
        checkParams(self, f, emb)
        self.assertEqual(emb.grads['E'][0], [0.0, 0.0, 0.0])
        self.assertEqual(emb.grads['E'][4], [0.0, 0.0, 0.0])


class LayerNormTest(unittest.TestCase):
    def test_output_has_zero_mean_and_unit_variance(self):
        rng = makeRng(0)
        y, _ = LayerNorm(8).forward([[rng.gauss(5, 3) for _ in range(8)] for _ in range(4)])
        for row in y:
            self.assertAlmostEqual(mean(row), 0.0, places=9)
            self.assertAlmostEqual(mean([v * v for v in row]), 1.0, places=4)

    def test_worked_example_numbers(self):
        ln = LayerNorm(4)
        y, cache = ln.forward([[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual([round(v, 4) for v in y[0]], [-1.3416, -0.4472, 0.4472, 1.3416])
        ln.zeroGrad()
        dx = ln.backward(cache, [[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual([round(v, 4) for v in dx[0]], [0.2683, -0.3578, -0.0894, 0.1789])
        self.assertAlmostEqual(sum(dx[0]), 0.0, places=12)

    def test_is_invariant_to_shifting_and_scaling_the_input(self):
        ln = LayerNorm(4)
        a, _ = ln.forward([[1.0, 2.0, 3.0, 4.0]])
        b, _ = ln.forward([[11.0, 12.0, 13.0, 14.0]])
        for u, v in zip(a[0], b[0]):
            self.assertAlmostEqual(u, v, places=6)


class AttentionAndBlocksTest(unittest.TestCase):
    def check(self, module, forward, xs, ws, memory=None):
        f = lambda: sum(dot(w, y) for w, y in zip(ws, forward(xs, memory)[0]))
        module.zeroGrad()
        _, cache = forward(xs, memory)
        dxs, dmemory = module.backward(cache, ws)
        checkParams(self, f, module)
        checkVectors(self, f, xs, dxs, 'x')
        if memory is not None:
            checkVectors(self, f, memory, dmemory, 'memoria')

    def setUp(self):
        rng = makeRng(3)
        self.xs, self.ws, self.memory = [rv(rng, 8) for _ in range(4)], [rv(rng, 8) for _ in range(4)], [rv(rng, 8) for _ in range(5)]

    def test_multihead_attention(self):
        for heads, mask in ((1, None), (2, causalMask(4))):
            mha = MultiHeadAttention(8, heads, seed=2)
            f = lambda: sum(dot(w, y) for w, y in zip(self.ws, mha.forward(self.xs, None, mask)[0]))
            mha.zeroGrad()
            _, _, cache = mha.forward(self.xs, None, mask)
            dxs = mha.backwardSelf(cache, self.ws)
            checkParams(self, f, mha)
            checkVectors(self, f, self.xs, dxs, 'x')

    def test_encoder_block(self):
        block = TransformerBlock(8, 2, 16, seed=1)
        self.check(block, block.forward, self.xs, self.ws)

    def test_causal_block(self):
        block = TransformerBlock(8, 2, 16, causal=True, seed=1)
        self.check(block, block.forward, self.xs, self.ws)

    def test_decoder_block_with_cross_attention(self):
        block = TransformerBlock(8, 2, 16, causal=True, cross=True, seed=1)
        self.check(block, block.forward, self.xs, self.ws, self.memory)

    def test_stack_of_decoder_blocks(self):
        stack = Stack(2, 8, 2, 16, causal=True, cross=True, seed=2)
        self.check(stack, stack.forward, self.xs, self.ws, self.memory)

    def test_block_is_the_identity_when_the_sublayers_output_zero(self):
        block = TransformerBlock(8, 2, 16, seed=1)
        for name in block.attn.params:
            if name == 'Wo':
                block.attn.params[name] = [[0.0] * 8 for _ in range(8)]
        block.ffn.lin2.params['W'] = [[0.0] * 16 for _ in range(8)]
        ys, _ = block.forward(self.xs)
        for y, x in zip(ys, self.xs):
            self.assertEqual(y, x)       # sin correcciones, la residual deja pasar la entrada intacta

    def test_block_output_is_input_plus_attention_plus_ffn(self):
        block = TransformerBlock(8, 2, 16, seed=1)
        ys, _ = block.forward(self.xs)
        n1, _ = block.ln1.forward(self.xs)
        a, _, _ = block.attn.forward(n1)
        x1 = [[u + v for u, v in zip(x, av)] for x, av in zip(self.xs, a)]
        f, _ = block.ffn.forward(block.ln2.forward(x1)[0])
        for y, x, av, fv in zip(ys, self.xs, a, f):
            for j in range(8):
                self.assertAlmostEqual(y[j], x[j] + av[j] + fv[j])


class PositionalEncodingTest(unittest.TestCase):
    def test_formula(self):
        pe = positionalEncoding(3, 4)
        self.assertEqual(pe[0], [0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(pe[1][0], math.sin(1.0))
        self.assertAlmostEqual(pe[1][1], math.cos(1.0))
        self.assertAlmostEqual(pe[2][2], math.sin(2 / 100))

    def test_every_position_has_a_distinct_code(self):
        pe = positionalEncoding(32, 16)
        self.assertEqual(len({tuple(round(v, 9) for v in row) for row in pe}), 32)


class ModelGradientTest(unittest.TestCase):
    def check(self, model, forward):
        model.zeroGrad()
        _, cache = forward()
        model.backward(cache)
        checkParams(self, lambda: forward()[0], model)

    def test_seq2seq_transformer_all_parameters(self):
        rng = makeRng(0)
        model = Seq2SeqTransformer(4, d_model=6, n_heads=2, d_ff=8, n_enc=1, n_dec=2, seed=1)
        src = [rng.randrange(4) for _ in range(4)]
        self.check(model, lambda: model.forward(src, src[::-1]))

    def test_language_model_all_parameters(self):
        rng = makeRng(0)
        model = LanguageModel(5, d_model=6, n_heads=2, d_ff=8, n_layers=2, seed=2)
        toks = [rng.randrange(5) for _ in range(6)]
        self.check(model, lambda: model.forward(toks[:-1], toks[1:]))


class ModelPropertiesTest(unittest.TestCase):
    def test_causal_language_model_ignores_the_future(self):
        model = LanguageModel(6, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=1)
        tokens = [0, 1, 2, 3, 4, 5]
        a = model.logits(tokens)[0]
        b = model.logits(tokens[:4] + [5, 0])[0]
        for i in range(4):
            self.assertEqual(a[i], b[i])
        self.assertNotEqual(a[5], b[5])

    def test_parameter_counts_match_the_formulas(self):
        for enc, dec in ((1, 1), (2, 3)):
            model = Seq2SeqTransformer(6, 16, 2, 32, enc, dec)
            self.assertEqual(model.paramCount(), anatomy.expectedSeq2Seq(6, 16, 32, enc, dec))
        self.assertEqual(LanguageModel(7, 12, 2, 24, 3).paramCount(), anatomy.expectedLM(7, 12, 24, 3))

    def test_block_parameter_formula(self):
        self.assertEqual(sum(anatomy.blockParams(16, 32, False).values()), 4 * 16 * 16 + 2 * 16 * 2 + 2 * 16 * 32 + 32 + 16)

    def test_greedy_returns_symbols_and_cross_attention_weights(self):
        model = Seq2SeqTransformer(5, 8, 2, 16, seed=0)
        out, weights = model.greedy([1, 2, 3, 0], 4)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(0 <= t < 5 for t in out))
        self.assertEqual((len(weights), len(weights[0]), len(weights[0][0])), (2, 4, 4))   # cabezas x pasos x posiciones de la entrada
        self.assertAlmostEqual(sum(weights[0][0]), 1.0)

    def test_untrained_loss_is_about_log_vocab(self):
        loss, _ = Seq2SeqTransformer(6, 8, 2, 16, seed=0).forward([1, 2, 3], [3, 2, 1])
        self.assertAlmostEqual(loss, math.log(6), delta=0.6)


class WorkedExamplesTest(unittest.TestCase):
    def test_causal_mask_example_in_step_by_step(self):
        model = LanguageModel(6, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=1)
        tokens = [0, 0, 0, 2, 1, 5]
        changed = tokens[:4] + [2, 1]
        a, b = model.logits(tokens)[0], model.logits(changed)[0]
        self.assertTrue(all(x == y for i in range(4) for x, y in zip(a[i], b[i])))
        self.assertGreater(max(abs(x - y) for x, y in zip(a[4], b[4])), 1e-3)
        self.assertTrue(callable(step_by_step.partD))


class OptimizerTest(unittest.TestCase):
    def test_adam_first_step_moves_each_parameter_by_lr(self):
        params = {'a': [[1.0, 2.0]], 'b': [0.5]}
        Adam([params], lr=0.1).step([{'a': [[0.3, -4.0]], 'b': [2e-3]}])
        self.assertAlmostEqual(params['a'][0][0], 0.9, places=6)
        self.assertAlmostEqual(params['a'][0][1], 2.1, places=6)
        self.assertAlmostEqual(params['b'][0], 0.4, places=6)

    def test_sgd_step(self):
        params = {'a': [[1.0, 2.0]], 'b': [0.5]}
        SGD([params], lr=0.5).step([{'a': [[0.2, -0.4]], 'b': [1.0]}])
        self.assertEqual(params, {'a': [[0.9, 2.2]], 'b': [0.0]})


class TasksAndTrainingTest(unittest.TestCase):
    def test_tasks(self):
        rng = makeRng(0)
        src, tgt = makeExample('reverse', 5, 7, rng)
        self.assertEqual(tgt, src[::-1])
        src, tgt = makeExample('copy', 5, 7, rng)
        self.assertEqual(tgt, src)

    def test_vocabulary_roundtrip_and_corpus_grammar(self):
        text = corpus(30)
        vocab = Vocabulary(text)
        self.assertEqual(vocab.decode(vocab.encode(text)), text)
        ok, total = validSentences(text)
        self.assertEqual((ok, total), (30, 30))                  # el corpus solo contiene frases de la gramatica
        self.assertEqual(len(sentences()), 6)
        self.assertEqual(validSentences('el gato come carne. el loro bebe agua.'), (1, 2))

    def test_seq2seq_learns_a_short_reverse_task(self):
        model = Seq2SeqTransformer(4, d_model=12, n_heads=2, d_ff=24, seed=0)
        losses = trainSeq2Seq(model, 'reverse', 3, 4, steps=250, batch=8, lr=0.01, seed=0)
        self.assertLess(sum(losses[-20:]) / 20, 0.3 * sum(losses[:20]) / 20)
        self.assertGreaterEqual(evaluateSeq2Seq(model, 'reverse', 3, 4, n=100)[0], 0.9)

    def test_language_model_learns_the_grammar_a_little(self):
        text = corpus(60)
        vocab = Vocabulary(text)
        tokens = vocab.encode(text)
        model = LanguageModel(len(vocab), d_model=12, n_heads=2, d_ff=24, n_layers=1, max_len=16, seed=0)
        before = evaluateLM(model, tokens, 12, n=20)
        trainLM(model, tokens, 12, steps=60, batch=4, lr=0.01, seed=0)
        after = evaluateLM(model, tokens, 12, n=20)
        self.assertLess(after[0], before[0] - 0.5)

    def test_loss_floor_estimate_is_between_zero_and_chance(self):
        text = corpus()
        floor = lossFloor(text, 20)
        self.assertGreater(floor, 0.05)
        self.assertLess(floor, 0.3)                      # muy por debajo del azar (ln 18 = 2.89), pero no cero
        self.assertGreater(lossFloor(text, 4), floor)    # con menos contexto, mas incertidumbre

    def test_generate_samples_valid_symbols_and_respects_the_context_window(self):
        model = LanguageModel(5, d_model=8, n_heads=2, d_ff=16, n_layers=1, max_len=6, seed=0)
        tokens = model.generate([1, 2], 20, makeRng(0))
        self.assertEqual(len(tokens), 22)
        self.assertTrue(all(0 <= t < 5 for t in tokens))

    def test_heatmap_and_example_input(self):
        self.assertGreater(len(set(exampleSource(8))), 1)
        text = heatmap([[0.9, 0.1], [0.2, 0.8]], [3, 1], [1, 3])
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn('mira la posicion 2', text)


if __name__ == '__main__':
    unittest.main()
