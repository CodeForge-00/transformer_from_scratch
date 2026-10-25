"""Entrenamiento y evaluacion de los dos modelos."""
from transformer.mathlib import argmax, makeRng, paramNorm, scaleGrads
from transformer.models import crossEntropy
from transformer.optim import Adam
from transformer.tasks import makeExample, makeSet


def _step(model, optimizer, pairs, n, clip):
    """Promedia los gradientes acumulados del lote, los recorta por norma global y da un paso de Adam."""
    grads = [g for _, g in pairs]
    for g in grads:
        scaleGrads(g, 1.0 / n)
    norm = paramNorm(*grads)
    if norm > clip:
        for g in grads:
            scaleGrads(g, clip / norm)
    optimizer.step(grads)


def trainSeq2Seq(model, task, length, vocab, steps=500, batch=8, lr=0.005, clip=1.0, seed=0, on_step=None):
    """Adam sobre lotes de secuencias nuevas. Devuelve la lista de perdidas medias por paso."""
    pairs = model.paramPairs()
    optimizer = Adam([p for p, _ in pairs], lr=lr)
    rng, losses = makeRng(seed), []
    for step in range(steps):
        model.zeroGrad()
        total = 0.0
        for _ in range(batch):
            src, tgt = makeExample(task, length, vocab, rng)
            loss, cache = model.forward(src, tgt)
            model.backward(cache)
            total += loss / batch
        _step(model, optimizer, pairs, batch, clip)
        losses.append(total)
        if on_step:
            on_step(step, total)
    return losses


def evaluateSeq2Seq(model, task, length, vocab, n=200, seed=999):
    """(exactitud por simbolo, exactitud por secuencia completa) con generacion voraz sobre secuencias nuevas."""
    tokens = sequences = 0
    for src, tgt in makeSet(task, length, vocab, n, makeRng(seed)):
        out, _ = model.greedy(src, length)
        correct = sum(a == b for a, b in zip(out, tgt))
        tokens += correct
        sequences += int(correct == length)
    return tokens / (n * length), sequences / n


def trainLM(model, tokens, context, steps=500, batch=8, lr=0.005, clip=1.0, seed=0, on_step=None):
    """Cada paso toma `batch` trozos de `context` simbolos del texto y aprende a predecir el simbolo siguiente en cada posicion."""
    pairs = model.paramPairs()
    optimizer = Adam([p for p, _ in pairs], lr=lr)
    rng, losses = makeRng(seed), []
    for step in range(steps):
        model.zeroGrad()
        total = 0.0
        for _ in range(batch):
            start = rng.randrange(len(tokens) - context - 1)
            chunk = tokens[start:start + context + 1]
            loss, cache = model.forward(chunk[:-1], chunk[1:])
            model.backward(cache)
            total += loss / batch
        _step(model, optimizer, pairs, batch, clip)
        losses.append(total)
        if on_step:
            on_step(step, total)
    return losses


def evaluateLM(model, tokens, context, n=100, seed=999):
    """(perdida media, exactitud del simbolo siguiente) sobre trozos al azar del texto, sin entrenar."""
    rng = makeRng(seed)
    loss_sum = correct = total = 0
    for _ in range(n):
        start = rng.randrange(len(tokens) - context - 1)
        chunk = tokens[start:start + context + 1]
        logits, _ = model.logits(chunk[:-1])
        loss, _ = crossEntropy(logits, chunk[1:])
        loss_sum += loss
        for z, t in zip(logits, chunk[1:]):
            correct += int(argmax(z) == t)
            total += 1
    return loss_sum / n, correct / total
