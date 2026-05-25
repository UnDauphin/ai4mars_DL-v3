# 00 — Justificación General del Proyecto: Segmentación Semántica de Terreno Marciano

---

## 1. Contexto y definición del problema

### El dataset AI4MARS

AI4MARS es un dataset público desarrollado conjuntamente por el Jet Propulsion Laboratory (JPL) y la NASA, construido a partir de imágenes capturadas por la cámara de navegación (NavCam) del rover Curiosity durante la misión Mars Science Laboratory (MSL). El dataset comprende más de 23.000 pares imagen-máscara etiquetados mediante una campaña de crowdsourcing que reunió más de 326.000 contribuciones de voluntarios, con un subconjunto de evaluación —el llamado *gold test set*— validado por científicos del JPL con acuerdo del 100% entre al menos tres anotadores especializados.

Cada máscara semántica asigna a cada píxel una de cuatro categorías de terreno:

| ID | Clase | Descripción |
|---|---|---|
| 0 | `soil` | Suelo firme, clase mayoritaria |
| 1 | `bedrock` | Roca base expuesta |
| 2 | `sand` | Arena suelta |
| 3 | `big_rock` | Rocas sueltas individuales, clase minoritaria |
| 255 | `ignore` | Zonas sin etiqueta válida, excluidas de la evaluación |

El objetivo de segmentación semántica es asignar una de estas etiquetas a cada píxel de la imagen de forma automática, produciendo un mapa denso que describe la composición del terreno frente al rover.

### Por qué es difícil este problema

Las imágenes NavCam presentan condiciones visuales poco habituales que las diferencian de los benchmarks estándar de segmentación semántica. Se trata de imágenes en escala de grises —convertidas a RGB para compatibilidad con los modelos, pero con los tres canales idénticos (media ≈ 0.23)— capturadas en un entorno sin agua, sin vegetación, sin fauna y sin estructuras artificiales. La variabilidad inter-clase es visualmente reducida: suelo, roca base y arena comparten texturas y tonalidades similares bajo iluminación solar variable y a distintas distancias del rover. Los bordes entre clases son frecuentemente gradientes continuos, no fronteras abruptas, lo que hace que la localización exacta de las transiciones dependa en gran medida de la interpretación del anotador.

A estas dificultades intrínsecas del dominio se suma el hecho de que el dataset fue construido mediante crowdsourcing: las etiquetas en las zonas de mayor ambigüedad son el resultado del consenso entre múltiples anotadores no especializados, con los errores y la variabilidad subjetiva que ello implica. El valor `ignore=255` refleja precisamente las zonas donde el consenso no fue suficiente para asignar una etiqueta fiable.

---

## 2. Relevancia e impacto del proyecto

### 2.1 Apoyo a la navegación autónoma de rovers

La motivación central de AI4MARS es operacional: un rover en la superficie de Marte no puede detenerse a esperar instrucciones cada vez que necesita decidir por dónde avanzar. La latencia de comunicación entre Marte y la Tierra varía entre 4 y 24 minutos según la posición orbital, lo que hace imposible el control manual en tiempo real para maniobras de detalle. La navegación autónoma —o, al menos, la supervisión semi-autónoma— requiere que el rover pueda interpretar el terreno frente a él y tomar decisiones de ruta sin intervención humana inmediata.

Un error de clasificación con consecuencias severas sería, por ejemplo, confundir arena suelta con suelo firme: el rover Spirit de la misión MER quedó atrapado irreversiblemente en 2009 al hundirse en una zona de arena no detectada como tal. La segmentación precisa del terreno es, en ese sentido, una precondición de seguridad para la operación del rover y para la integridad de la misión.

### 2.2 Apoyo a la ciencia geológica

Más allá de la navegación, la capacidad de clasificar automáticamente el tipo de terreno tiene valor científico directo. Las imágenes NavCam son uno de los instrumentos de documentación más prolíficos de la misión MSL: Curiosity ha capturado decenas de miles de imágenes a lo largo de sus más de 12 años en Marte. Analizar manualmente ese volumen de imágenes para caracterizar la geología del recorrido es una tarea inabordable para un equipo pequeño de científicos. Un sistema de segmentación automática fiable permitiría catalogar sistemáticamente los tipos de superficie atravesados, identificar zonas de interés geológico (exposiciones de bedrock, acumulaciones de arena, presencia de rocas) y correlacionar esa información con otros instrumentos del rover (espectrómetro, cámara química, etc.).

### 2.3 Transferibilidad a otras misiones y dominios

El problema de segmentar terreno en imágenes de exploración robotizada no es exclusivo de Marte. Misiones lunares actuales y planificadas (Artemis, Chang'e, LUPEX), así como aplicaciones de robótica terrestre en entornos sin estructura (zonas de desastre, minería subterránea, exploración submarina), comparten características similares: bajo contraste visual, ausencia de referencias artificiales, iluminación variable y necesidad de tomar decisiones de navegación en tiempo real. Los métodos y las lecciones aprendidas en AI4MARS son, en principio, transferibles a cualquiera de estos dominios.

---

## 3. Dificultades técnicas específicas de AI4MARS

La tabla siguiente resume los principales factores que hacen de AI4MARS un problema de segmentación particularmente exigente:

| Factor | Descripción | Impacto |
|---|---|---|
| **Desbalance extremo** | `big_rock` = 0.89% de los píxeles | Métricas globales engañosas; clases difíciles subrepresentadas |
| **Ambigüedad visual** | Texturas similares entre clases adyacentes | Alta tasa de confusión en bordes |
| **Etiquetas ruidosas** | Crowdsourcing con variabilidad inter-anotador | GT no es un estándar perfecto |
| **Dominio fuera de distribución** | Imágenes en escala de grises, sin referencias familiares | Modelos preentrenados en ImageNet necesitan adaptación |
| **Resolución reducida** | Imágenes redimensionadas a 256×256 px | Pérdida de detalles finos, especialmente en objetos pequeños |
| **Zonas `ignore`** | Píxeles con valor 255, excluidos de métricas | Requieren manejo explícito en la implementación |
| **Variabilidad de iluminación** | Ángulo solar cambiante, sombras | Texturas inconsistentes para la misma clase |

El caso de `big_rock` merece atención especial. Esta clase no solo es la más minoritaria —con menos del 1% de los píxeles en el conjunto de entrenamiento y un peso compensatorio de 3.58× en la función de pérdida— sino que es también la más crítica para la navegación: una roca suelta de tamaño relevante puede bloquear el avance del rover o dañar sus ruedas si no es detectada. El hecho de que sea simultáneamente la clase más importante operacionalmente y la más difícil de detectar por su escasez define el cuello de botella principal del problema.

---

## 4. Valor metodológico de la comparación de modelos

### 4.1 Por qué no basta con una sola arquitectura

Comparar múltiples arquitecturas sobre el mismo problema tiene valor científico y práctico que va más allá de encontrar "el mejor modelo". Distintas familias de modelos tienen ventajas estructurales diferentes: los modelos CNN puros (DeepLabV3+, MarsSeg) destacan en la captura de texturas locales y son más estables con datasets pequeños; los modelos Transformer puros (SegFormer, DepthFormer) tienen mayor campo receptivo efectivo y capturan contexto global de largo alcance; los modelos híbridos (TerSeg) intentan combinar ambas propiedades a costa de mayor complejidad.

Evaluar todos estos enfoques sobre el mismo dataset, con el mismo split, las mismas métricas y el mismo protocolo de evaluación permite identificar qué características arquitectónicas son realmente determinantes para este problema específico, y no solo en términos de mIoU global sino en la detección de cada clase individualmente.

### 4.2 Por qué el análisis por clase es esencial

El mIoU global es la métrica estándar de referencia en segmentación semántica, pero en un dataset tan desbalanceado como AI4MARS puede ocultar diferencias importantes. Un modelo que detecta soil y bedrock con IoU > 0.97 pero falla completamente en `big_rock` puede reportar un mIoU aparentemente aceptable, ocultando que la clase más crítica para la navegación está siendo ignorada. El análisis por clase expone exactamente ese tipo de comportamiento y permite comparar los modelos con un criterio más completo.

### 4.3 Por qué el protocolo experimental importa

El proyecto implementa un protocolo de evaluación diseñado para maximizar la comparabilidad y la reproducibilidad:

- **Split fijo**: un único archivo `split_indices_msl6k.pkl` compartido entre todos los modelos garantiza que todos son entrenados y validados sobre exactamente los mismos datos.
- **Gold test externo**: las 322 imágenes del gold test (min3-100agree, JPL) nunca entran al entrenamiento ni a la validación. Esto reproduce el estándar de evaluación de la literatura, permitiendo comparar con trabajos previos.
- **Múltiples seeds**: cada modelo se entrena con 3 semillas independientes. La media y la desviación estándar del mIoU permiten distinguir rendimiento real de varianza de inicialización.
- **Métricas por clase**: IoU individual para soil, bedrock, sand y big_rock, además de pixel accuracy y mIoU global.
- **Exclusión de `ignore`**: los píxeles con valor 255 se excluyen tanto de la función de pérdida durante el entrenamiento como del cálculo de métricas en evaluación, mediante `ignore_index=255`.

Este protocolo hace que los resultados sean sólidos, comparables entre modelos y entre este proyecto y la literatura relevante.

---

## 5. Aporte de la propuesta original

### 5.1 Cómo se entiende la originalidad en investigación aplicada

En el campo de la visión por computador, una contribución original no requiere necesariamente inventar una arquitectura nueva desde cero. La investigación aplicada —en la que se adapta un método de referencia a un dominio específico, se evalúa rigurosamente su comportamiento, y se analizan sus ventajas y limitaciones en ese contexto— es una forma legítima y valiosa de generar conocimiento, especialmente cuando el dominio de aplicación tiene características visuales y de distribución muy distintas de los benchmarks estándar sobre los que los modelos fueron originalmente desarrollados.

En este proyecto, la propuesta original consiste en la adaptación y evaluación de SegFormer-B2 al dominio AI4MARS. La búsqueda sistemática realizada en Web of Science no encontró ningún trabajo previo que aplique esta arquitectura a la segmentación de terreno marciano ni al dataset AI4MARS en particular, entre los 50 artículos recuperados. La contribución es, por tanto, original en el sentido relevante para este trabajo: aporta evidencia nueva sobre el comportamiento de una arquitectura establecida en un dominio no explorado.

### 5.2 Qué aporta concretamente la propuesta

La elección de SegFormer-B2 no es arbitraria. Su encoder jerárquico multiescala (MiT-B2) captura información a resoluciones complementarias, lo que es relevante para detectar tanto estructuras extensas (bedrock, suelo) como objetos pequeños y dispersos (`big_rock`). La ausencia de codificación posicional fija lo hace robusto a variaciones de resolución. Su decoder MLP ligero reduce el coste computacional en inferencia. Y con solo 27.35M parámetros, es el modelo más compacto entre los de alto rendimiento en este benchmark.

Los resultados empíricos respaldan la elección: mIoU de 0.8337 en el gold test, segunda posición en el ranking general, mejor pixel accuracy (0.9814), mejor rendimiento en bedrock y sand, y la menor varianza entre seeds (σ = 0.0063) de todos los modelos evaluados. Esto posiciona a SegFormer-B2 como la opción más eficiente en la frontera rendimiento–complejidad del proyecto.

---

## 6. Impacto esperado del proyecto

### Impacto científico

El proyecto produce evidencia empírica comparada sobre el rendimiento de cinco arquitecturas de segmentación semántica en un dominio de exploración planetaria real. Este tipo de benchmark riguroso —con gold test externo, múltiples seeds y métricas por clase— es el que la comunidad científica necesita para decidir qué enfoques priorizar en futuras investigaciones. La identificación de `big_rock` como cuello de botella universal (todos los modelos muestran IoU < 0.52 en esta clase) delimita con precisión el problema abierto más importante del dominio.

### Impacto metodológico

El proyecto demuestra que arquitecturas de propósito general, desarrolladas para benchmarks terrestres (Cityscapes, ADE20K), son adaptables al dominio marciano con resultados competitivos y sin modificaciones arquitectónicas profundas. Esto reduce la barrera de entrada para futuras investigaciones sobre AI4MARS, que pueden tomar este benchmark como línea base.

### Impacto computacional

La comparación explícita entre modelos en términos de parámetros, tiempo de entrenamiento y rendimiento (Figura: Precisión vs. Complejidad y Precisión vs. Costo Computacional) ofrece una guía práctica para seleccionar arquitecturas según las restricciones del entorno de despliegue. Para sistemas embarcados en rovers —donde los recursos son limitados— el resultado de que SegFormer-B2 con 27.35M parámetros alcanza un rendimiento estadísticamente comparable al mejor modelo (49.19M parámetros) es directamente relevante.

### Transferibilidad

Los métodos, el protocolo de evaluación y las conclusiones sobre las dificultades del dominio son transferibles a otros problemas de segmentación en entornos con características similares: imágenes de baja textura, clases desbalanceadas, etiquetas ruidosas y necesidad de navegación autónoma en entornos no estructurados.

---

## 7. Conclusión

Este proyecto aborda un problema con consecuencias operacionales directas en misiones de exploración planetaria: la clasificación automática y precisa del tipo de terreno en imágenes capturadas por rovers en la superficie de Marte. La dificultad del problema —combinación de desbalance extremo de clases, ambigüedad visual intrínseca al dominio marciano, etiquetado ruidoso por crowdsourcing y una clase minoritaria de alta criticidad operacional— justifica tanto el estudio riguroso como la comparación sistemática de múltiples enfoques arquitectónicos.

El valor del proyecto no reside en un único resultado numérico, sino en la calidad del protocolo experimental, la amplitud de la comparación, y la evidencia generada sobre qué funciona y qué no funciona en este dominio específico. La propuesta de SegFormer-B2 como modelo original —una arquitectura de referencia no aplicada previamente a AI4MARS, con fundamento técnico sólido y resultados empíricos competitivos— añade una contribución concreta y defendible al análisis del terreno marciano con aprendizaje profundo.
