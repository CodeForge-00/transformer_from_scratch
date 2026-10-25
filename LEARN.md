# Transformers desde cero, solo con matemática

Guía para aprender **cómo funciona un transformer**: de qué piezas se compone, qué hace cada una y cómo se entrena, sin que ninguna
biblioteca esconda la lógica. Todo el código usa únicamente listas de Python y el módulo `math`: cada normalización, cada residual y
cada derivada está escrita a la vista. Es lento a propósito (sirve para entender, no para entrenar modelos grandes).

Este proyecto **supone la atención ya entendida** (consulta, clave, valor, softmax, varias cabezas, máscaras). Si no, empieza por
[`attention_from_scratch`](https://github.com/CodeForge-00/attention_from_scratch). Aquí se construye lo que la rodea hasta tener dos
transformers completos: uno **codificador-decodificador** (el del artículo original) y uno **solo-decodificador** (la forma de GPT), y se
entrenan con tareas diminutas.

## 0. Cómo estudiarlo

1. Lee las secciones 1 a 7 (qué añade un transformer a la atención, pieza por pieza).
2. Ejecuta `python -m transformer.step_by_step` y `python -m transformer.anatomy` y sigue los números.
3. Lee la sección 8 junto al código de `blocks.py` y `models.py`: cada `backward` lleva la ecuación que implementa.
4. Ejecuta los experimentos de la sección 9 y compara con lo medido aquí.
5. Haz los ejercicios de la sección 10.

| Archivo | Para qué sirve |
|---|---|
| `transformer/mathlib.py` | Las operaciones (producto matriz-vector, softmax y su derivada…) con listas y `math`. |
| `transformer/module.py` | `Linear`, `LayerNorm`, `Embedding` y `FeedForward`, con su `backward`. |
| `transformer/attention.py` | La atención de varias cabezas (autoatención, cruzada y máscaras), ahora con gradientes acumulados. |
| `transformer/blocks.py` | La codificación de posiciones, el **bloque** (residuales + LayerNorm + atención + FFN) y la pila de bloques. |
| `transformer/models.py` | `Seq2SeqTransformer` (codificador-decodificador) y `LanguageModel` (solo-decodificador). |
| `transformer/optim.py`, `training.py`, `tasks.py` | Adam y SGD a mano, los bucles de entrenamiento y los datos. |
| `transformer/step_by_step.py` | Ejemplos numéricos: LayerNorm, posiciones, un bloque y la máscara causal. |
| `transformer/anatomy.py` | Las piezas y sus parámetros, con la fórmula que cuadra el total. |
| `transformer/reverse_task.py` | Experimento: invertir secuencias (la misma tarea que con GRU). |
| `transformer/char_lm.py` | Experimento: un modelo de lenguaje diminuto que escribe texto. |
| `tests/test_transformer.py` | Gradientes contra derivadas numéricas y los números de este documento. |

**Notación.** Los vectores son columnas y `W x` es el producto de la matriz `W` por el vector `x`. `⊙` es el producto elemento a elemento,
`ᵀ` la traspuesta, `d` (o `d_model`) la dimensión de los vectores de cada posición y `T` la longitud de la secuencia.

---

## 1. De la atención al transformer: qué falta

La atención por sí sola es una operación: dado un conjunto de vectores, cada posición construye una **mezcla ponderada** de los demás. Para
convertirla en un modelo que aprenda hay que resolver cuatro cosas:

1. **La atención no ve el orden** (es equivariante a permutaciones): hay que decirle a cada posición dónde está → *codificación de posiciones*.
2. **La atención solo mezcla**; no transforma cada vector por separado → una *red densa por posición* (FFN).
3. **Una red profunda se vuelve difícil de entrenar** (gradientes que se desvanecen, escalas que se descontrolan) → *conexiones residuales* y *normalización por capas*.
4. **Hay que apilar y conectar**: varios bloques seguidos, y según el modelo, una máscara causal y atención cruzada.

El resultado, de entrada a salida:

```
símbolos ─► embedding + posición ─► [ Bloque ] × N ─► LayerNorm ─► Linear ─► softmax ─► probabilidad de cada símbolo

Bloque:    x ─► x + Atención( LN(x) ) ─► x' + FFN( LN(x') ) ─► y          (con LN = LayerNorm; cada «+» es una residual)
```

---

## 2. Embedding y codificación de posiciones

**Embedding.** Cada símbolo del vocabulario tiene un vector `E[t]` de `d` números, que se aprende: es una tabla `vocabulario × d`. Solo las
filas usadas reciben gradiente (`dE[t] += dx`).

**Posiciones.** Como la atención no ve el orden (ver `attention_from_scratch`, sección 6.6), se **suma** a cada embedding un vector que
depende de la posición. Aquí se usa el sinusoidal del artículo original, fijo y sin parámetros:

```
PE[pos][2i]   = sin( pos / 10000^{2i/d} )          PE[pos][2i+1] = cos( pos / 10000^{2i/d} )
```

`python -m transformer.step_by_step`, parte B (`d = 4`):

| posición | componente 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 0,0000 | 1,0000 | 0,0000 | 1,0000 |
| 1 | 0,8415 | 0,5403 | 0,0100 | 1,0000 |
| 2 | 0,9093 | −0,4161 | 0,0200 | 0,9998 |
| 3 | 0,1411 | −0,9900 | 0,0300 | 0,9996 |

Las dos primeras componentes giran rápido (periodo `2π`) y las dos últimas despacio (periodo `2π · 100`): como las manecillas de un reloj,
juntas dan un «código» distinto a cada posición. Los valores quedan entre −1 y 1. (El artículo original señala además que el vector de una
posición `pos + k` es una función lineal del de `pos`, lo que facilitaría aprender a mirar «k posiciones atrás»; lo cito de memoria.)

La entrada de la primera capa es `x_i = E[símbolo_i] + PE[i]`. La posición **no se entrena**: su gradiente se descarta.

---

## 3. Normalización por capas (LayerNorm)

Para cada vector `x` de `d` componentes, por separado:

```
μ = media(x)        σ² = media( (x − μ)² )        x̂ = (x − μ) / √(σ² + ε)        y = γ ⊙ x̂ + β
```

`x̂` tiene media 0 y varianza 1, sin depender del resto de la secuencia ni del lote (a diferencia de la normalización por lotes). `γ` y
`β` son parámetros que le devuelven a la red la libertad de escalar y desplazar. Mantiene las magnitudes controladas capa tras capa.

`step_by_step`, parte A, con `x = [1, 2, 3, 4]`:

```
μ = 2,5      σ² = ((−1,5)² + (−0,5)² + 0,5² + 1,5²)/4 = 1,25      σ = 1,1180
x̂ = [ −1,3416 , −0,4472 , 0,4472 , 1,3416 ]
```

### Su retropropagación

`x̂` depende de `x` por tres caminos: directamente, y a través de `μ` y de `σ²` (que también dependen de `x`). Derivar los tres da

```
dγ += dy ⊙ x̂           dβ += dy           dx̂ = dy ⊙ γ
dx = (1/σ) · ( dx̂ − media(dx̂) − x̂ ⊙ media(dx̂ ⊙ x̂) )
```

El segundo término viene de `μ` (se resta el promedio) y el tercero de `σ²` (se resta la parte paralela a `x̂`). Con `dy = [1, 0, 0, 0]`:

```
dx̂ = [1, 0, 0, 0]     media(dx̂) = 0,25     media(dx̂ ⊙ x̂) = −1,3416/4 = −0,3354
dx = (1/1,118) · ( [1,0,0,0] − 0,25 − x̂ · (−0,3354) ) = [ 0,2683 , −0,3578 , −0,0894 , 0,1789 ]
```

Y **suman 0**: sumar una constante a todo `x` no cambia `x̂`, así que ningún reparto de `dx` puede «subirlo todo a la vez». Un test
(`test_worked_example_numbers`) fija estos números.

---

## 4. Conexiones residuales

Una conexión residual **suma la entrada a la salida de la subcapa**:

```
y = x + subcapa(x)          dy/dx = I + d(subcapa)/dx
```

La derivada tiene un `I` (la identidad): el gradiente de la salida llega a la entrada **tal cual por el camino directo**, y además por la
subcapa. Es el mismo camino aditivo que la celda de la LSTM (`c_t = f ⊙ c_{t-1} + …`): por eso se pueden apilar muchos bloques sin que el
gradiente se desvanezca. Y cambia la forma de ver la red: el vector de cada posición es un **flujo residual** al que cada subcapa solo
**agrega una corrección**.

`step_by_step`, parte C: un bloque sobre 3 vectores con pesos aleatorios. Para la posición 1:

```
x          = [−0,7313 ,  0,6949 ,  0,5275 , −0,4899]
a          = [−0,3981 , −0,3892 , −0,4735 ,  0,1270]       lo que propone sumar la atención
x1 = x + a = [−1,1294 ,  0,3057 ,  0,0540 , −0,3629]
f          = [ 0,0211 ,  0,0742 , −0,0318 , −0,1844]       lo que propone sumar la FFN
y = x1 + f = [−1,1083 ,  0,3799 ,  0,0222 , −0,5473]
```

Un test comprueba que `y = x + a + f` exactamente, y que con las salidas de las subcapas puestas a cero el bloque devuelve la entrada intacta.

**Dónde va la LayerNorm.** El artículo original normaliza *después* de sumar (`LN(x + subcapa(x))`, «post-LN»). Este proyecto usa la
variante **pre-LN** de GPT-2, que normaliza *antes* de cada subcapa (`x + subcapa(LN(x))`): deja el camino residual sin tocar y suele
entrenar con más estabilidad. Por eso hace falta una LayerNorm final tras el último bloque. (Lo cito de memoria; conviene contrastarlo.)

---

## 5. La red densa por posición (FFN)

```
FFN(x) = W₂ · relu( W₁ x + b₁ ) + b₂          W₁: d_ff × d      W₂: d × d_ff
```

Se aplica **igual y por separado a cada posición**: mismos pesos, sin mirar a las demás. El reparto de trabajo es:

- la **atención** mezcla información *entre* posiciones;
- la **FFN** transforma cada vector *por dentro*, con una no linealidad.

Sin FFN, apilar atenciones solo daría mezclas de mezclas (lineales en los valores). El `d_ff` suele ser unas cuatro veces `d`; aquí es el doble para que sea pequeño. Derivada de la ReLU: 1 donde la entrada era positiva y 0 en el resto.

---

## 6. El bloque completo

**Bloque de codificador** (o de un modelo solo-decodificador, con máscara causal):

```
x1 = x  + Autoatención( LN₁(x) )
y  = x1 + FFN( LN₂(x1) )
```

**Bloque de decodificador** de un modelo codificador-decodificador:

```
x1 = x  + Autoatención CAUSAL( LN₁(x) )              cada posición solo mira hacia atrás
x2 = x1 + Atención CRUZADA( LN₂(x1), memoria )       las consultas vienen del decodificador, las claves y valores del codificador
y  = x2 + FFN( LN₃(x2) )
```

### Parámetros

| Pieza | Parámetros |
|---|---|
| Atención (`W_q, W_k, W_v, W_o`, sin sesgos) | `4 d²` |
| LayerNorm (`γ, β`) | `2 d` |
| FFN | `2 d·d_ff + d_ff + d` |
| Bloque de codificador | `4d² + 2·2d + FFN` |
| Bloque de decodificador | `2·4d² + 3·2d + FFN` |

`python -m transformer.anatomy` los calcula para cualquier configuración y comprueba que coinciden con los del modelo (con `d = 16`,
`d_ff = 32`: un bloque de codificador tiene 2160, uno de decodificador 3216, y el modelo completo de 1 + 1 bloques, 5750).

### Coste

Cada cabeza calcula `T × T` puntuaciones por capa: el coste de la atención crece con el **cuadrado** de la longitud (64 puntuaciones con
`T = 8`, unos 16,8 millones con `T = 4096`). Es la razón por la que las secuencias largas son caras.

---

## 7. Los dos modelos

### 7.1 Codificador-decodificador (el del artículo «Attention Is All You Need»)

```
entrada ─► embedding + posición ─► CODIFICADOR (N bloques, autoatención sin máscara) ─► memoria
salida anterior (BOS + símbolos previos) ─► embedding + posición ─► DECODIFICADOR (N bloques: causal + cruzada con la memoria + FFN) ─► Linear ─► softmax
```

- El **codificador** lee toda la entrada a la vez; cada posición ve todas las demás. Produce la *memoria*.
- El **decodificador** genera la salida símbolo a símbolo. Su autoatención es **causal** y su atención **cruzada** consulta la memoria.
- Se entrena con *teacher forcing*: el decodificador recibe `BOS` y los símbolos **correctos** anteriores, y todas las posiciones se calculan a la vez (la máscara causal impide hacer trampa).
- Al **generar**, se parte de `BOS`, se elige el símbolo más probable, se añade a la entrada y se repite. Aquí se recalcula el decodificador entero en cada paso (simple, no eficiente).

### 7.2 Solo-decodificador (la forma de GPT)

```
símbolos ─► embedding + posición ─► N bloques (autoatención CAUSAL + FFN) ─► LayerNorm ─► Linear ─► softmax
```

Sin codificador ni atención cruzada. Aprende a **predecir el símbolo siguiente** a partir de los anteriores: en cada posición `i` la
salida es la distribución de `símbolo_{i+1}`, así que de un trozo de `n + 1` símbolos salen `n` ejemplos de entrenamiento. Para
**generar texto** se parte de un prefijo, se muestrea el siguiente de `softmax(logits / temperatura)`, se añade y se repite.

La **temperatura** controla cuánto arriesga: cerca de 0, siempre el más probable; con 1, según las probabilidades del modelo; mayor que 1, más ruido.

Los modelos de lenguaje actuales de tipo GPT son esencialmente esta arquitectura, mucho más grande, con más datos y con otras mejoras. (Lo que
usa cada familia —solo codificador, solo decodificador o ambos— lo cito de memoria.)

---

## 8. Retropropagación del bloque

Se propaga hacia atrás en el orden inverso del `forward`, con las reglas de las secciones anteriores y las de la atención:

```
y = x2 + FFN(LN₂(x2))
   dx2 = dy                                         ← camino directo de la residual
       + LN₂.backward( FFN.backward(dy) )            ← camino por la subcapa
x2 = x1 + CruzadaAtención(LN_c(x1), memoria)       (solo en el decodificador)
   dx1 = dx2 + LN_c.backward( dq )                  y   dmemoria += dkv
x1 = x + Autoatención(LN₁(x))
   dx  = dx1 + LN₁.backward( dx_atención )
```

En cada residual el gradiente tiene **dos caminos que se suman**: el directo (la parte `I`) y el de la subcapa. La memoria del codificador
recibe gradiente de **todos** los bloques del decodificador, así que se acumula. Cada módulo guarda en `self.grads` lo que le corresponde
y devuelve el gradiente de su entrada.

### Verificación numérica

`tests/test_transformer.py` compara cada `backward` con la derivada numérica (`dL/dθ ≈ (L(θ+ε) − L(θ−ε)) / 2ε`, `ε = 10⁻⁶`) para
**todos** los parámetros y entradas de `Linear`, `LayerNorm`, `FeedForward`, `Embedding`, la atención, los tres tipos de bloque, una pila de
dos bloques de decodificador y los dos modelos completos (1252 y 633 parámetros). El error real es del orden de `10⁻⁹`.

---

## 9. Experimentos medidos

### 9.1 Invertir una secuencia: transformer frente a GRU con atención

La misma tarea que en `attention_from_scratch`: invertir una secuencia de un vocabulario de 6 símbolos. Aquí, un codificador-decodificador con
`d_model = 16`, 2 cabezas, `d_ff = 32` y **un bloque por lado**: **5750 parámetros**. 1000 pasos de Adam (tasa 0,005, lotes de 8, gradiente recortado a
norma 1), **3 semillas**, y 200 secuencias nuevas para evaluar, generando de forma voraz.

`python -m transformer.reverse_task --lengths 4 8 12 16 --steps 1000 --seeds 3` (unos 25 minutos en Python puro). Cada celda es *exactitud por símbolo / exactitud por
secuencia completa*; el azar acertaría un 17 % de los símbolos.

| Longitud T | Transformer (3 semillas, 1000 pasos) | GRU sin atención | GRU con producto escalar | GRU con atención aditiva |
|---|---|---|---|---|
| 4 | **100 % / 100 %** (las 3) | 98 % / 94 % | 99 % / 98 % | 100 % / 100 % |
| 8 | **100 % / 100 %** (las 3) | 84 % / 52 % | 76 % / 30 % | 100 % / 99 % |
| 12 | **100 % / 100 %** (las 3) | 62 % / 4 % | 57 % / 2 % | 89 % / 66 % |
| 16 | **100 % / 100 %** (las 3) | sin medir | sin medir | sin medir |
| Parámetros | 5750 | 2606 | 2702 | 3230 |

(Las columnas de GRU son de `attention_from_scratch`: **una semilla** y 2500 pasos.)

Qué se ve:

- **Las 12 ejecuciones del transformer** (4 longitudes × 3 semillas) acertaron el 100 % de las secuencias completas, con menos pasos que los modelos de GRU.
- **Donde las GRU se degradan, el transformer no:** con `T = 12`, la mejor GRU (atención aditiva) acierta el 66 % de las secuencias completas y el transformer, el 100 %. No se aprecia deterioro hasta `T = 16`, la mayor longitud probada.
- **Adónde mira.** La atención cruzada dibuja una antidiagonal limpia: al generar el primer símbolo mira la última posición de la entrada, y así hasta la primera (ver el mapa de calor en el `README.md`). Las dos cabezas coinciden.
- **Una explicación probable, sin comprobar:** la codificación de posiciones da a cada posición un «código», así que la atención cruzada puede pedir «la posición `T − t`» directamente. En la GRU, esa información de posición tiene que descubrirla el propio estado recurrente. Además, la atención conecta cualquier par de posiciones en un solo paso, sin cadena de multiplicaciones.

**Cautelas.** La comparación **no es de igual tamaño ni de igual presupuesto**: el transformer tiene 1,8 veces los parámetros de la mejor GRU y se entrenó 1000 pasos frente a 2500; las cifras
de las GRU son de una sola semilla y no se midió `T = 16`. Y es una tarea muy favorable a la atención con posiciones (mirar a una posición fija relativa al paso). No dice nada sobre
modelos reales ni sobre tareas donde lo difícil sea otra cosa que localizar una posición.


### 9.2 Un modelo de lenguaje diminuto

Un modelo solo-decodificador de 2 bloques (`d_model = 24`, 2 cabezas, `d_ff = 48`, contexto de 20 caracteres, **10 482 parámetros**) aprende a predecir el
carácter siguiente en un texto de 2338 caracteres (18 símbolos distintos) hecho con una gramática de seis frases: `el gato come pez.`, `el gato bebe leche.`,
`el perro come carne.`, `el perro bebe agua.`, `el loro come maiz.` y `el loro bebe agua.`. Hay algo que aprender: tras `el gato come ` solo puede venir `pez`.

`python -m transformer.char_lm --steps 1500 --sample-every 250` (unos 9 minutos en Python puro). **Una semilla**, 1500 pasos de Adam (tasa 0,005, lotes de 8).
Cada cierto número de pasos escribe una muestra a partir del prefijo `el `, con temperatura 0,8:

| Paso | Pérdida | Muestra (primeros caracteres) |
|---|---|---|
| 0 | 2,89 (azar) | `el tohcheocgireapoctzoraprml.gmtzepeoi ngrm gr.cpcieuph alt hlg` |
| 250 | 0,214 | `el perro come carne. el perro come carne. el perro bebe agua. el perrro bebe agua.` |
| 750 | 0,185 | `el perro come carne. el perro come carne. el perro bebe agua. el perro come carne.` |
| 1250 | 0,181 | `el loro come maiz. el perro bebe agua. el l loro bebe agua. el perro come carne.` |
| 1500 | 0,185 | `el perro come carne. el loro bebe agua. el gato come pez. el loro come maiz.` |

Con 600 caracteres generados con otra semilla: **28 de 30 frases completas (93 %) pertenecen a la gramática.** Acierta el 91 % de los caracteres siguientes.

Qué se ve:

- **Pasa de ruido a frases con sentido, poco a poco.** Con 250 pasos (90 s) ya escribe palabras y la estructura `el X come/bebe Y.`, pero con poca variedad (solo `perro`); con 1250 a 1500 mezcla los tres animales y los empareja bien (`gato` con `pez`, `perro` con `carne` o `agua`, `loro` con `maiz` o `agua`).
- **La pérdida llega casi al mínimo posible.** Quedó en 0,18 y el mínimo estimado es 0,16 (el cálculo está en `lossFloor`). **El mínimo no es cero**: en cada trozo de 20 caracteres, las primeras posiciones ven muy poca historia y no se pueden predecir bien (con 1 carácter de historia cuesta 1,12 nats; con más de 11, 0,08). El 91 % de aciertos tampoco puede llegar al 100 %.

### Un error mío que enseña algo

Mis primeras ejecuciones, con tres variantes y 2500 pasos, daban **solo un 2–4 % de frases válidas**, con una pérdida de 0,18 que parecía buena. Todas tenían el mismo síntoma: la
**primera frase salía perfecta** (`el perro come carne.`) y a continuación el texto se degradaba (`el el l l perrrrrro`). La causa: entrené con un contexto de 20
caracteres (posiciones 0 a 19), pero generaba con una ventana de 24, así que a partir del carácter 21 el modelo recibía **posiciones que nunca había visto**. Con una
ventana igual al contexto, la misma arquitectura pasó del 4 % al 93 % de frases válidas.

Dos lecciones:

1. **Una pérdida baja no garantiza que la generación funcione.** La pérdida solo miraba las posiciones 0 a 19; la generación usaba otras. Hay que comprobar el procedimiento real de uso, no solo la métrica de entrenamiento.
2. **Mi primer cálculo del mínimo de la pérdida estaba mal** (dije ~0,09): olvidé que las primeras posiciones de cada trozo tienen poca historia. Lo detecté al contar bien y comparar con el modelo.

**Cautelas.** Es **una sola semilla** y una gramática de seis frases: el modelo prácticamente aprende a recitar frases fijas, no lenguaje general. Con otro texto, otra
temperatura u otro contexto los resultados cambian. No dice nada sobre modelos de lenguaje reales.


---

## 10. Ejercicios

1. **A mano.** Calcula la LayerNorm de `x = [2, 4, 6, 8]`. ¿Qué relación tiene con la de `[1, 2, 3, 4]`? Compruébalo con `LayerNorm(4).forward`.
2. **Sin residual.** En `TransformerBlock.forward`, quita la suma `x + a` (usa solo `a`). ¿Cómo cambia la pérdida tras 250 pasos de `reverse_task`? ¿Y el gradiente que llega a las primeras capas?
3. **Post-LN.** Cambia el bloque a `LN(x + subcapa(x))`. ¿Qué cambia en el `backward`? La prueba de gradiente te dirá si acertaste. ¿Entrena igual de bien?
4. **Sin posiciones.** En `Seq2SeqTransformer._embed`, no sumes `PE`. Predice qué pasará con la tarea de invertir y comprueba (pista: la autoatención no ve el orden).
5. **Sin FFN.** Quita la FFN de los bloques. ¿Sigue aprendiendo la inversión? ¿Y el modelo de lenguaje?
6. **Cabezas.** Cambia `--heads` a 1, 2 y 4 en `reverse_task`. ¿Cambia la exactitud? Mira en `--show` si las cabezas aprenden la misma atención.
7. **Temperatura.** Genera con `--temperature 0.2`, `1` y `2` en `char_lm`. ¿Qué pasa con la proporción de frases válidas?
8. **Capas.** Prueba `--layers 1` y `3` en `char_lm`. ¿Merece la pena una capa más con tan poco texto?
9. **Derívala tú.** Deriva `dx` de la LayerNorm con los tres caminos (`x̂`, `μ`, `σ²`) y compáralo con el código.
10. **Una variante.** Implementa **GELU** en lugar de ReLU en la FFN, con su derivada y su `backward`. El test de gradiente comprobará tu derivada.

---

## 11. Límites y qué falta para un modelo de lenguaje real

- Es **Python puro**: mucho más lento que NumPy o una biblioteca de aprendizaje automático. Las tareas y los modelos son diminutos a propósito (miles de parámetros, no miles de millones).
- Los resultados dependen de la semilla, la tarea y el presupuesto de entrenamiento: las cifras de este documento son de esta implementación y estos tamaños, y no dicen nada sobre modelos reales.
- **No hay** dropout ni otra regularización, ni *weight decay*, ni calentamiento de la tasa de aprendizaje, ni precisión mixta.
- **Tokenización:** aquí cada carácter o símbolo es un token. Los modelos reales usan subpalabras (por ejemplo BPE).
- **Generación eficiente:** se recalcula toda la secuencia en cada paso; los modelos reales guardan las claves y valores ya calculados (*KV cache*).
- **Escala y entrenamiento posterior:** datos, cómputo y ajuste fino con preferencias humanas, que explican gran parte de lo que hace útil a un modelo real, no están aquí.
- Los autores y años de la bibliografía están citados de memoria: conviene contrastarlos con los artículos originales.

## 12. Referencias

- A. Vaswani et al., «Attention Is All You Need», 2017: el transformer, la atención escalada, las varias cabezas y la codificación sinusoidal.
- J. Ba, J. Kiros y G. Hinton, «Layer Normalization», 2016.
- K. He et al., «Deep Residual Learning for Image Recognition», 2015: las conexiones residuales.
- A. Radford et al., «Language Models are Unsupervised Multitask Learners» (GPT-2), 2019: el modelo solo-decodificador y la variante con normalización previa.
- D. Bahdanau, K. Cho y Y. Bengio, 2014, y M.-T. Luong et al., 2015: la atención en modelos con RNN (ver `attention_from_scratch`).
- D. Kingma y J. Ba, «Adam: A Method for Stochastic Optimization», 2014.
