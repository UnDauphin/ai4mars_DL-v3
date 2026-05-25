# 01 — Justificación de SegFormer-B2 como Modelo Original del Proyecto AI4MARS


## 1. Sentido de la "originalidad" en este proyecto

En el contexto de este entregable, la originalidad de SegFormer-B2 no se refiere a que la arquitectura sea nueva en la literatura general —fue publicada en NeurIPS 2021 (Xie et al.)— sino a que su **aplicación al dominio específico de AI4MARS y a la segmentación semántica de terreno marciano con imágenes NavCam** no aparece documentada en los trabajos revisados.

La búsqueda sistemática realizada en bases de datos académicas Q1 (Scopus, Web of Science, IEEE Xplore, arXiv) identificó publicaciones sobre SegFormer en dominios muy diversos. Para contextualizar la amplitud de su adopción, la Tabla 1 resume los dominios de aplicación encontrados en los 50 artículos recuperados de Web of Science:

**Tabla 1. Dominios de aplicación de SegFormer en los 50 artículos de Web of Science**

| Dominio de aplicación | Ejemplos representativos |
|---|---|
| Geociencias y riesgo natural | Detección de deslizamientos, cambio de uso de suelo |
| Teledetección y SAR | Imágenes de satélite, ondas oceánicas internas, espacios verdes urbanos |
| Agricultura | Enfermedades de plantas, impurezas en grano de arroz, rodenticidas |
| Medicina e imagen clínica | Tumores cerebrales, pólipos de colon, hígado, cuello uterino, aorta |
| Industria y materiales | Corrosión metálica, fugas magnéticas, carbón y roca |
| Imagen submarina | Imágenes subacuáticas, segmentación de montes submarinos |
| Visión UAV y minería | Minas desde UAV, extracción de vegetación |
| Segmentación de celdas | Detección de fallas en manufactura |

Ninguno de estos 50 trabajos corresponde a la segmentación de terreno marciano, al dataset AI4MARS, ni a imágenes NavCam o HazCam de rovers de la NASA. En ese sentido, la adaptación y evaluación de SegFormer-B2 sobre AI4MARS constituye una **contribución original dentro del alcance de este proyecto**: una arquitectura de referencia aplicada a un dominio donde no había sido evaluada en la bibliografía revisada.

---

## 2. Justificación técnica de SegFormer para la segmentación de terreno marciano

### 2.1 Encoder jerárquico multiescala (Mix Transformer — MiT)

SegFormer emplea un encoder Transformer jerárquico denominado MiT (Mix Transformer), que genera representaciones en cuatro escalas de resolución: 1/4, 1/8, 1/16 y 1/32 de la imagen original. A diferencia de ViT, que produce un único mapa de características de baja resolución, MiT captura simultáneamente detalles finos de alta resolución (bordes, texturas locales) y contexto semántico global (relaciones de largo alcance entre regiones).

Esta propiedad es directamente relevante para las imágenes NavCam del rover Curiosity. El terreno marciano presenta una mezcla de estructuras a escalas muy distintas: superficies extensas de suelo y bedrock que requieren contexto amplio para ser correctamente delimitadas, y rocas sueltas (`big_rock`) que pueden ocupar regiones pequeñas y dispersas dentro de la imagen. Un encoder de escala única tiende a sacrificar uno de los dos extremos; el diseño jerárquico de MiT los aborda de forma complementaria.

### 2.2 Ausencia de codificación posicional fija

Una limitación conocida de los Transformers clásicos (ViT, SETR) es su dependencia de una codificación posicional de resolución fija. Cuando la resolución de inferencia difiere de la de entrenamiento —situación habitual en segmentación semántica, donde las imágenes del mundo real tienen tamaños variables— el modelo debe interpolar los embeddings posicionales, lo que degrada el rendimiento de forma medible.

SegFormer reemplaza la codificación posicional explícita por **Mix-FFN**, un módulo que incorpora una convolución 3×3 dentro de la red feed-forward de cada bloque Transformer. Xie et al. (2021) demostraron que esta convolución es suficiente para proporcionar información posicional implícita, eliminando la sensibilidad a cambios de resolución. El beneficio cuantificado en Cityscapes es notable: con codificación posicional fija la caída de mIoU al cambiar de resolución es de 3.3 puntos; con Mix-FFN la caída se reduce a 0.7 puntos.

En el contexto de AI4MARS, las imágenes NavCam del dataset original presentan resoluciones heterogéneas, y el proyecto estandariza a 256×256 px mediante preprocesamiento. La robustez ante variaciones de resolución reduce el riesgo de degradación en eventuales ajustes de tamaño o en el uso de otras cámaras del rover.

### 2.3 Decoder MLP ligero y campo receptivo efectivo amplio

El decoder de SegFormer consiste exclusivamente en capas MLP lineales que unifican los canales de las cuatro escalas del encoder, los upsamplean a 1/4 de la imagen y producen la máscara de predicción final. Esta simplicidad no es una concesión en rendimiento: es viable precisamente porque el encoder Transformer ya tiene un **campo receptivo efectivo (ERF)** mucho mayor que encoders CNN equivalentes.

Xie et al. (2021) visualizan explícitamente este fenómeno: el decoder MLP de SegFormer exhibe atención tanto local como altamente no-local al mismo tiempo, al combinar las representaciones de los cuatro estadios del encoder. El mismo decoder acoplado a un backbone CNN (ResNet, ResNeXt) produce resultados significativamente inferiores, lo que confirma que el beneficio es propio de la combinación encoder Transformer + decoder MLP.

Para la segmentación marciana, esto se traduce en dos ventajas concretas: mayor estabilidad durante el entrenamiento (menos hiperparámetros a ajustar en el decoder) y menor coste computacional en inferencia, relevante para sistemas de visión embarcados donde los recursos son limitados.

---

## 3. Relación con las características específicas del problema AI4MARS

### 3.1 Detección de `big_rock`: clase minoritaria y difícil

El análisis de pesos de clase del proyecto confirma que `big_rock` representa únicamente el **0.89% de los píxeles** en el conjunto de entrenamiento, con un peso compensatorio de 3.58× en la función de pérdida. Esta es la clase más difícil de todos los modelos evaluados, y también la que más diferencia a los modelos entre sí.

La capacidad de SegFormer para capturar contexto global (estadio 4 del encoder, con atención altamente no-local) es especialmente útil para detectar objetos pequeños y dispersos. Modelos con campo receptivo limitado —como los basados en convoluciones atrous sin mecanismo de atención global— pueden perder el contexto necesario para distinguir una roca suelta del bedrock circundante. SegFormer-B2 alcanza un IoU de `big_rock` de **0.459** en el gold test, superando a DeepLabV3+ (0.408) y siendo el segundo mejor resultado general, solo por detrás de TerSeg (0.510).

### 3.2 Representación multiescala y texturas marcianas

Las imágenes NavCam del rover Curiosity (convertidas a RGB desde escala de grises, con media≈0.23 en los tres canales) presentan bajo contraste inter-clase y alta ambigüedad en bordes. La representación multiescala del MiT permite al modelo operar simultáneamente sobre detalles texturales finos (que ayudan a distinguir arena de suelo) y sobre formas globales (que ayudan a delimitar regiones extensas de bedrock). Esta combinación es difícil de lograr con encoders de escala única.

### 3.3 Estabilidad en entrenamiento con datos limitados

El proyecto entrena sobre un subconjunto de 6.000 imágenes MSL (4.200 train / 1.800 val) con 3 seeds independientes por modelo. SegFormer-B2 muestra la **menor varianza entre todos los modelos evaluados** (σ = 0.0063), lo que indica que su rendimiento es consistente y reproducible. Esta estabilidad es valiosa en escenarios de investigación donde el tiempo de cómputo limita el número de repeticiones posibles.

---

## 4. Justificación científica y computacional integrada

| Dimensión | SegFormer-B2 | Relevancia para AI4MARS |
|---|---|---|
| **Parámetros** | 27.35M | El modelo más compacto entre los de alto rendimiento |
| **Tiempo de entrenamiento** | ~71 min/run | Razonable; DeepLabV3+ requiere ~154 min con peor resultado |
| **mIoU (gold test)** | 0.8337 ± 0.0063 | 2° en ranking; diferencia con 1° (0.8381) dentro del IC95 |
| **Pixel accuracy** | 0.9814 | La más alta de todos los modelos (incluyendo TerSeg) |
| **IoU big_rock** | 0.459 | 2° en ranking; supera a modelos CNN más pesados |
| **Varianza entre seeds** | σ = 0.0063 | La más baja; indica robustez y reproducibilidad |
| **Robustez a resolución** | Mix-FFN sin PE | Ventaja estructural documentada en el paper original |

La figura de eficiencia computacional del proyecto (precisión vs. complejidad y vs. costo de entrenamiento) muestra que SegFormer-B2 logra un mIoU comparable al mejor modelo (TerSeg) con aproximadamente la **mitad de los parámetros** (27.35M vs. 49.19M) y sin necesitar arquitectura dual-branch ni módulos de fusión personalizados. En el eje de costo computacional, SegFormer-B2 se posiciona en la zona de alto rendimiento con tiempo de entrenamiento moderado, mientras que DeepLabV3+ —con menor mIoU— requiere más del doble de tiempo.

Desde la perspectiva de la escalabilidad, la familia SegFormer cubre un rango amplio (B0 a B5), lo que permite ajustar el modelo al hardware disponible sin cambiar la arquitectura base. SegFormer-B5 alcanza 84.0% mIoU en Cityscapes con menos parámetros que SETR, lo que evidencia su potencial de escalar si se dispone de mayor capacidad de cómputo.

---

## 5. Comparación breve con los benchmarks del proyecto

La siguiente tabla resume los resultados del gold test (322 imágenes MSL, min3-100agree):

| Modelo | mIoU (mean ± std) | IoU soil | IoU bedrock | IoU sand | IoU big_rock | Params (M) | Train (min) |
|---|---|---|---|---|---|---|---|
| **TerSeg** | **0.8381 ± 0.0297** | 0.981 | 0.919 | 0.943 | **0.510** | 49.19 | ~50 |
| **SegFormer-B2** | **0.8337 ± 0.0063** | 0.978 | **0.940** | **0.958** | 0.459 | **27.35** | ~71 |
| DeepLabV3+ | 0.8183 ± 0.0198 | 0.975 | 0.939 | 0.952 | 0.408 | 42.00 | ~154 |
| DepthFormer-RGB | 0.7219 ± 0.0290 | 0.941 | 0.896 | 0.898 | 0.154 | 42.02 | ~2* |
| MarsSeg | 0.7054 ± 0.0318 | 0.914 | 0.871 | 0.863 | 0.173 | 34.87 | ~4* |

*DepthFormer y MarsSeg muestran tiempos muy bajos; inferencia: posiblemente entrenados con `FAST_SUBSET=True` o con menos epochs, lo que podría explicar también sus mIoU más bajos. No se afirma esto como causa definitiva.

**Observaciones relevantes:**

- La diferencia en mIoU entre TerSeg y SegFormer-B2 es de 0.0044, valor que cae dentro del IC95 de ambos modelos (±0.0336 y ±0.0071 respectivamente). Estadísticamente, no puede afirmarse que TerSeg sea superior con certeza.
- SegFormer-B2 supera a TerSeg en bedrock (0.940 vs. 0.919) y sand (0.958 vs. 0.943), y tiene la mayor pixel accuracy (0.9814 vs. 0.9775).
- TerSeg supera a SegFormer-B2 en `big_rock` (0.510 vs. 0.459), lo que se explica por su arquitectura dual-branch CNN+Transformer, diseñada específicamente para capturar tanto detalles locales como contexto global de forma independiente y luego fusionarlos.
- SegFormer-B2 logra resultados comparables a TerSeg con 22M menos de parámetros, sin arquitectura especializada para el dominio marciano.

---

## 6. Conclusión para defensa oral

SegFormer-B2 se justifica como modelo original de este proyecto por tres razones complementarias. Primero, no existen trabajos publicados —en los 50 artículos revisados de Web of Science— que apliquen SegFormer al dominio AI4MARS o a la segmentación de terreno marciano, lo que otorga novedad en la aplicación. Segundo, sus características arquitectónicas —encoder jerárquico multiescala, ausencia de codificación posicional fija, decoder MLP ligero— responden de forma directa a los retos específicos de este problema: imágenes de bajo contraste, clases desbalanceadas, necesidad de capturar objetos a múltiples escalas y robustez ante variaciones de resolución. Tercero, los resultados empíricos confirman que es el segundo modelo en mIoU global (0.8337, estadísticamente indistinguible del primero en IC95), el mejor en dos de las cuatro clases, el más estable entre seeds, y el más eficiente en términos de parámetros entre los modelos de alto rendimiento. Esta combinación de fundamento teórico, evidencia empírica y ausencia de aplicación previa en el dominio hace a SegFormer-B2 defendible como la contribución técnica central del proyecto.
