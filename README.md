# transformer_from_scratch

Aprender **cómo funciona un transformer**, escrito desde cero y solo con matemática: Python puro, con listas y el módulo `math` (sin NumPy
ni otras bibliotecas), para que ninguna biblioteca esconda la lógica. Cada normalización, cada residual y cada retropropagación está
escrita a mano y verificada contra derivadas numéricas.

Es el paso siguiente a [`attention_from_scratch`](https://github.com/CodeForge-00/attention_from_scratch): allí se explica la atención;
aquí se construye lo que la rodea (posiciones, normalización por capas, conexiones residuales, red densa por posición) hasta tener dos
transformers completos que se entrenan:

- un **codificador-decodificador**, el del artículo original «Attention Is All You Need», que aprende a invertir secuencias;
- un **modelo de lenguaje solo-decodificador**, la forma de GPT en miniatura, que aprende a escribir texto.

**Empieza por [`LEARN.md`](LEARN.md)**: el tutorial, con ejemplos numéricos, derivaciones y ejercicios.

## Qué contiene

| Archivo | Contenido |
|---|---|
| `LEARN.md` | El tutorial: qué añade un transformer a la atención, pieza por pieza, y cómo se entrena. |
| `transformer/module.py` | `Linear`, `LayerNorm`, `Embedding` y `FeedForward`, con `forward` y `backward`. |
| `transformer/attention.py` | La atención de varias cabezas (autoatención, atención cruzada, máscaras). |
| `transformer/blocks.py` | La codificación de posiciones, el **bloque** del transformer y la pila de bloques. |
| `transformer/models.py` | `Seq2SeqTransformer` (codificador-decodificador) y `LanguageModel` (solo-decodificador). |
| `transformer/optim.py`, `training.py`, `tasks.py` | Adam y SGD con sus fórmulas a la vista, los bucles de entrenamiento y los datos. |
| `transformer/step_by_step.py` | Ejemplos numéricos: LayerNorm, posiciones, un bloque y la máscara causal. |
| `transformer/anatomy.py` | Las piezas, sus parámetros y la fórmula que cuadra el total. |
| `transformer/reverse_task.py` | Experimento: invertir secuencias con el codificador-decodificador. |
| `transformer/char_lm.py` | Experimento: un modelo de lenguaje diminuto que escribe frases. |
| `tests/` | Gradientes contra derivadas numéricas y los números del tutorial. |

## Cómo usar los ejemplos

No hay que instalar nada: es Python 3 puro. Todos los comandos se ejecutan **desde la carpeta del proyecto**:

```bash
cd transformer_from_scratch
python3 -m transformer.<nombre>        # añade -h para ver las opciones de cada uno
```

Orden recomendado, de lo más rápido a lo más lento:

| # | Ejemplo | Tarda | Qué te enseña |
|---|---|---|---|
| 1 | `step_by_step` | < 1 s | LayerNorm, posiciones, un bloque y la máscara causal, con números. |
| 2 | `anatomy` | < 1 s | De qué piezas se compone un transformer y cuántos parámetros tiene cada una. |
| 3 | `reverse_task` | de 1 a varios minutos | Que un transformer aprende a invertir secuencias, y adónde mira. |
| 4 | `char_lm` | varios minutos | Cómo un modelo de lenguaje pasa de caracteres al azar a frases con sentido. |
| 5 | los tests | unos 15 s | Que cada `backward` está bien. |

### 1. `step_by_step`: lo que añade un transformer, con números

```bash
python3 -m transformer.step_by_step
```

Cuatro partes: **A** la LayerNorm de `[1, 2, 3, 4]` y su retropropagación; **B** la codificación de posiciones; **C** un bloque entero sobre 3
vectores, mostrando cada paso del «flujo residual»; **D** la máscara causal.

```
A. LAYERNORM de x = [1, 2, 3, 4]
   x_hat = (x - mu) / sqrt(var + eps) = [-1.3416, -0.4472,  0.4472,  1.3416]
   Hacia atras, con dL/dy = [1, 0, 0, 0]:
     dL/dx = [ 0.2683, -0.3578, -0.0894,  0.1789]      suma = 0.0000

D. LA MASCARA CAUSAL: cambiar el futuro no cambia el pasado
   texto A = [0, 0, 0, 2, 1, 5]
   texto B = [0, 0, 0, 2, 2, 1]      (distinto solo en las posiciones 5 y 6)
   salida de la posicion 4: diferencia maxima entre A y B = 0.00e+00   <- no depende del futuro
   salida de la posicion 5: diferencia maxima entre A y B = 8.46e-01   <- cambia: ve el simbolo nuevo
```

**Cómo leerlo:** en A, la salida tiene media 0 y varianza 1, y el gradiente suma 0. En D, una diferencia **exactamente 0** significa que
una posición no puede ver el futuro: es lo que permite entrenar a predecir el símbolo siguiente sin hacer trampa.

### 2. `anatomy`: las piezas y sus parámetros

```bash
python3 -m transformer.anatomy
python3 -m transformer.anatomy --d-model 32 --heads 4 --d-ff 64 --enc 2 --dec 2 --vocab 20
```

Imprime los parámetros de cada bloque (atención `4d²`, LayerNorm `2d`, FFN `2·d·d_ff + d_ff + d`), de cada parte del modelo y el total, que
comprueba contra la fórmula. Al final muestra cómo crece el coste de la atención con la longitud.

```
  codificador (o solo-decodificador):  autoatencion 1024 + LayerNorm 1 32 + LayerNorm 2 32 + FFN 1072  =  2160
  TOTAL                                 5750     (formula: 5750)
```

### 3. `reverse_task`: el transformer invierte secuencias

```bash
python3 -m transformer.reverse_task --lengths 3 --steps 150 --show           # unos segundos
python3 -m transformer.reverse_task --lengths 4 8 12 --steps 1000 --seeds 3  # varios minutos
```

Entrena un codificador-decodificador para **invertir una secuencia** (`[4, 2, 5, 2] → [2, 5, 2, 4]`) y mide cuánto acierta con secuencias
nuevas. Es la misma tarea del proyecto de atención, así que los resultados se pueden comparar con los de la GRU.

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--task` | `reverse` | `reverse` invierte; `copy` copia. |
| `--lengths` | `4 8 12` | Longitudes a probar (una fila por longitud). |
| `--steps` | 1000 | Pasos de Adam (lotes de 8). |
| `--seeds` | 1 | Repite con varias semillas; se muestra la media y cada semilla. |
| `--d-model`, `--heads`, `--d-ff`, `--layers` | 16, 2, 32, 1 | Tamaño del transformer. |
| `--show` | no | Al final imprime el **mapa de calor** de la atención cruzada, una tabla por cabeza. |

Salida real de la primera línea (`--lengths 3 --steps 150 --show`, mapa de la cabeza 1):

```
Atencion cruzada, T=3, cabeza 1:
       entrada:  4  2  5
       posicion: 1  2  3
  genera 5 (paso  1):        @   mira la posicion 3
  genera 2 (paso  2):     *  :   mira la posicion 2
  genera 4 (paso  3):  @         mira la posicion 1
```

**Cómo leerlo:** cada celda de la tabla de resultados es `exactitud por símbolo / por secuencia completa` (el azar acertaría un 17 % de los
símbolos). En el mapa de calor, cada fila es un símbolo generado y cada columna una posición de la entrada; el carácter indica el peso
(` .:-=+*#%@`, de menos a más). Aquí el modelo mira primero la posición 3, luego la 2 y la 1: una **antidiagonal**, que es lo que exige invertir.

### 4. `char_lm`: un modelo de lenguaje que escribe frases

```bash
python3 -m transformer.char_lm --steps 500 --sample-every 100      # unos minutos
python3 -m transformer.char_lm --steps 2500 --sample-every 250     # el experimento de abajo
```

Entrena un modelo solo-decodificador sobre un texto de frases de una gramática pequeña (`el gato come pez.`, `el perro bebe agua.`…).
**Cada cierto número de pasos escribe una muestra**, para ver cómo pasa de caracteres al azar a frases con sentido, poco a poco. Al final
mide cuántas de las frases que genera pertenecen a la gramática.

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--steps` | 1500 | Pasos de entrenamiento. |
| `--sample-every` | 250 | Cada cuántos pasos escribe una muestra. |
| `--d-model`, `--heads`, `--d-ff`, `--layers` | 24, 2, 48, 2 | Tamaño del modelo. |
| `--context` | 20 | Cuántos caracteres ve para predecir el siguiente. |
| `--temperature` | 0.8 | Al generar: menor = más seguro y repetitivo, mayor = más arriesgado. |
| `--prompt` | `el ` | Con qué texto empieza a generar. |

### 5. Usar las clases desde tu código

```python
from transformer.models import Seq2SeqTransformer, LanguageModel
from transformer.training import trainSeq2Seq, evaluateSeq2Seq

# Codificador-decodificador: invertir secuencias de un vocabulario de 6 simbolos
model = Seq2SeqTransformer(vocab=6, d_model=16, n_heads=2, d_ff=32, n_enc=1, n_dec=1, max_len=16, seed=0)
trainSeq2Seq(model, 'reverse', length=4, vocab=6, steps=250)
print(evaluateSeq2Seq(model, 'reverse', 4, 6))            # (exactitud por simbolo, por secuencia completa)
salida, pesos = model.greedy([4, 2, 5, 2], 4)             # pesos[cabeza][paso][posicion de la entrada]

# Un forward y un backward a mano
model.zeroGrad()                                          # antes de cada lote: pone a cero los gradientes acumulados
loss, cache = model.forward([4, 2, 5, 2], [2, 5, 2, 4])
model.backward(cache)                                     # los gradientes quedan en model.paramPairs()

# Modelo de lenguaje (solo decodificador): predice el simbolo siguiente
lm = LanguageModel(vocab=18, d_model=24, n_heads=2, d_ff=48, n_layers=2, max_len=24)
loss, cache = lm.forward(tokens[:-1], tokens[1:])         # targets[i] es el simbolo que sigue a tokens[i]
tokens_nuevos = lm.generate(prefijo, 60, rng, temperature=0.8)
```

### 6. Los tests

```bash
python3 -m unittest discover -s tests -t .
```

Comprueban que cada `backward` (`Linear`, `LayerNorm`, `FeedForward`, `Embedding`, la atención, los tres tipos de bloque, una pila de bloques
y los dos modelos completos, todos sus parámetros) coincide con la derivada numérica, y fijan los números de los ejemplos del tutorial. Si
cambias una línea de un `backward` y te equivocas, fallan.

### Ideas para experimentar

- Quita la residual en `TransformerBlock.forward` y mira cómo cambia la pérdida.
- Cambia `--heads` en `reverse_task --show` y compara qué mira cada cabeza.
- Prueba `--temperature 0.2` y `2` en `char_lm` y mira cuántas frases son válidas.
- Más ejercicios en la sección 10 de [`LEARN.md`](LEARN.md).

## Resultados medidos

**Invertir una secuencia** (`python3 -m transformer.reverse_task --lengths 4 8 12 16 --steps 1000 --seeds 3`, unos 25 minutos). Codificador-decodificador de
**5750 parámetros**, 1000 pasos de Adam, 3 semillas. Cada celda: *exactitud por símbolo / por secuencia completa*, en 200 secuencias nuevas.

| Longitud T | Transformer (3 semillas) | GRU sin atención | GRU producto escalar | GRU atención aditiva |
|---|---|---|---|---|
| 4 | **100 % / 100 %** | 98 % / 94 % | 99 % / 98 % | 100 % / 100 % |
| 8 | **100 % / 100 %** | 84 % / 52 % | 76 % / 30 % | 100 % / 99 % |
| 12 | **100 % / 100 %** | 62 % / 4 % | 57 % / 2 % | 89 % / 66 % |
| 16 | **100 % / 100 %** | sin medir | sin medir | sin medir |

- Las 12 ejecuciones del transformer (4 longitudes × 3 semillas) acertaron todas las secuencias completas.
- **Cautela:** la comparación con las GRU (de [`attention_from_scratch`](https://github.com/CodeForge-00/attention_from_scratch)) **no es de igual tamaño ni presupuesto**: el
  transformer tiene 5750 parámetros frente a 2606–3230 y se entrenó 1000 pasos frente a 2500; las cifras de las GRU son de una sola semilla.

**Un modelo de lenguaje diminuto** (`python3 -m transformer.char_lm --steps 1500 --sample-every 250`, unos 9 minutos). Modelo solo-decodificador de 10 482 parámetros sobre
frases de una gramática de seis frases. Una semilla.

```
paso    0: el tohcheocgireapoctzoraprml.gmtzepeoi ngrm gr.cpcieuph alt hlg
paso  250: el perro come carne. el perro come carne. el perro bebe agua. el perrro bebe agua.
paso 1500: el perro come carne. el loro bebe agua. el gato come pez. el loro come maiz.
```

- La pérdida llega a 0,18, cerca del mínimo posible (≈ 0,16), y **el 93 % de las frases generadas (28 de 30) pertenecen a la gramática**.
- **Un error mío que enseña algo:** mis primeras ejecuciones daban solo un 2–4 % de frases válidas con la misma pérdida, porque generaba con una ventana de 24 posiciones y el
  modelo solo había visto 20. Con la ventana corregida pasó del 4 % al 93 %. Una pérdida baja no garantiza que la generación funcione (más en [`LEARN.md`](LEARN.md)).
- **Cautela:** una semilla y una gramática de seis frases; el modelo prácticamente aprende a recitar frases fijas.


## Límites

- Python puro: lento, y los modelos son diminutos a propósito (miles de parámetros, no miles de millones).
- Los resultados dependen de la semilla, la tarea y el presupuesto de entrenamiento; son de esta implementación y estos tamaños.
- Sin dropout, sin *weight decay*, sin calentamiento de la tasa de aprendizaje, sin tokenización por subpalabras y sin *KV cache*.
- Los autores y años de la bibliografía están citados de memoria: conviene contrastarlos con los artículos originales.
