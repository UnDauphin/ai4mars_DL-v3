# 02 — Análisis de la Discrepancia Visual entre Predicción y Ground Truth en AI4MARS


## 1. El problema de la inspección visual como criterio de evaluación

Una de las primeras reacciones al examinar las salidas de un modelo de segmentación es compararlas visualmente con la máscara ground truth (GT). Si la predicción "se ve diferente", la intuición inmediata es que el modelo ha fallado. Esta intuición, aunque comprensible, es técnicamente incorrecta como criterio de evaluación y puede llevar a conclusiones equivocadas.

En el caso de AI4MARS, la discrepancia visual entre la máscara predicha y la GT puede coexistir perfectamente con métricas cuantitativas sólidas, y esta coexistencia no es un defecto del modelo sino una consecuencia de fenómenos bien documentados en el campo de la segmentación semántica. A continuación se explican con fundamento técnico.

---

## 2. Factores que producen divergencia visual sin implicar mal desempeño

### 2.1 Desbalance extremo de clases

En el conjunto de entrenamiento de este proyecto, la distribución de píxeles es marcadamente asimétrica:

| Clase | Frecuencia real | Peso compensatorio |
|---|---|---|
| soil | 36.98% | 0.086× |
| bedrock | 50.32% | 0.063× |
| sand | 11.81% | 0.269× |
| big_rock | **0.89%** | **3.58×** |

Incluso con pesos por clase aplicados a la función de pérdida, el modelo recibe señal de gradiente mayoritariamente de las clases dominantes (soil y bedrock ocupan el 87.3% de los píxeles). Un modelo que clasifique correctamente toda la región de soil y bedrock y cometa errores en los fragmentos dispersos de `big_rock` puede tener una pixel accuracy superior al 97%, pero su máscara predicha parecerá visualmente muy diferente de la GT en las zonas donde las rocas fueron erróneamente omitidas o desplazadas. La métrica global no lo penaliza proporcionalmente porque esas regiones son una fracción minúscula del total de píxeles.

Esto no significa que el modelo esté mal: significa que el desbalance del dataset limita estructuralmente la detectabilidad de la clase minoritaria, independientemente de la arquitectura empleada.

### 2.2 Ambigüedad en bordes y zonas de transición

Las imágenes NavCam del rover Curiosity capturan superficies con iluminación lateral variable y sin color en muchos casos (las imágenes del proyecto tienen media ≈ 0.23 en los tres canales, lo que evidencia que son efectivamente grises convertidas a RGB). En estas condiciones, los límites entre clases adyacentes —por ejemplo, entre bedrock y soil, o entre sand y soil— son gradientes continuos, no fronteras nítidas.

El modelo aprende a trazar un límite binario donde existe una transición continua. Inevitablemente, ese límite difiere en algunos píxeles del que trazó el anotador. A nivel visual, un borde desplazado 5-10 píxeles produce una diferencia perceptible, pero su impacto sobre el IoU depende del tamaño relativo de la región: en regiones grandes (soil, bedrock), un error de borde afecta marginalmente al IoU; en regiones pequeñas (`big_rock`), un borde desplazado puede cambiar drásticamente la métrica.

### 2.3 Etiquetas ruidosas por crowdsourcing

El dataset AI4MARS fue etiquetado mediante crowdsourcing, con más de 326.000 etiquetas aportadas por voluntarios. Incluso el gold test set —322 imágenes validadas por científicos del JPL con acuerdo mínimo del 100% entre 3 anotadores— no está exento de subjetividad: la definición de qué píxel pertenece a `big_rock` vs. `bedrock`, o a `sand` vs. `soil`, tiene una componente interpretativa.

Cuando el modelo produce una predicción que difiere de la GT en zonas ambiguas, puede que el modelo no esté equivocado: puede estar reproduciendo la decisión que otro anotador igualmente calificado habría tomado. Esta es la razón por la cual el nivel de acuerdo entre anotadores (min3 vs. min1, min2) cambia sustancialmente las métricas de todos los modelos, sin que la capacidad del modelo haya variado. Una predicción visualmente discrepante respecto a una GT con ruido de etiquetado no es automáticamente una predicción errónea.

### 2.4 Resolución reducida y pérdida de detalle fino

Todas las imágenes del proyecto fueron redimensionadas a 256×256 píxeles mediante interpolación bilineal (imágenes) y nearest-neighbor (máscaras). Las imágenes originales del dataset AI4MARS tienen resoluciones variables y en muchos casos superiores. Este redimensionamiento comprime estructuras finas —como bordes de rocas pequeñas o fragmentos dispersos de arena— en pocos píxeles, aumentando la ambigüedad inherente.

Un modelo que predice una región de `big_rock` como una mancha ligeramente más pequeña o desplazada respecto a la GT puede estar operando dentro del margen de incertidumbre introducido por el propio proceso de preprocesamiento. La discrepancia visual en ese caso no es un fallo del modelo sino un artefacto del redimensionamiento.

### 2.5 Efecto de suavizado de la función de pérdida

La función de pérdida empleada —cross-entropy ponderada— penaliza la predicción incorrecta de cada píxel de forma independiente, pero optimiza el promedio sobre todo el batch. El resultado es que el modelo aprende a estar "correctamente calibrado en promedio", lo que en la práctica puede manifestarse como bordes ligeramente suavizados y regiones minoritarias con contornos más difusos que la GT.

Este fenómeno es estructural en cualquier segmentación supervisada con cross-entropy: el modelo minimiza la pérdida esperada, no maximiza la coincidencia exacta de cada píxel. Las máscaras predichas tienden a ser más "continuas" y menos ruidosas que las GT generadas por humanos, lo que produce diferencias visuales aunque la métrica sea alta.

### 2.6 El papel de `ignore_index=255`

En AI4MARS, los píxeles con valor 255 corresponden a zonas no etiquetadas que son **excluidas de la loss y de las métricas**. Esto tiene dos consecuencias visuales que pueden sorprender:

Primero, la GT que el anotador validó puede tener franjas completas en negro (ignore) donde el modelo simplemente no tiene ninguna señal de entrenamiento. Lo que el modelo predice en esas zonas es irrelevante para la métrica, pero es plenamente visible en la imagen. Una comparación visual sin distinguir qué píxeles son `ignore` lleva a juzgar como errores predicciones que el protocolo de evaluación ignora explícitamente.

Segundo, regiones de la imagen que visualmente parecen pertenecer a una clase (por ejemplo, suelo al fondo de la imagen) pueden estar marcadas como 255 porque el anotador consideró que la profundidad o el ángulo hacían la etiqueta incierta. El modelo puede predecir esa región con una clase razonable —y esa predicción se vería "diferente" de la GT— sin que ello afecte en absoluto a ninguna métrica.

### 2.7 Trade-off entre precisión global y IoU por clase

El IoU de una clase se define como la intersección entre píxeles predichos y píxeles verdaderos de esa clase, dividida entre su unión. Una máscara que visualmente "se parece mucho" a la GT puede tener un IoU bajo si el modelo erró sistemáticamente en la clase correcta de pequeñas regiones (falsos negativos concentrados en `big_rock`). Por el contrario, una máscara que luce diferente puede tener un IoU alto si los errores son píxeles aislados en bordes, que afectan marginalmente al numerador y denominador del cociente.

En otras palabras: **IoU no es percepción visual**. Mide solapamiento geométrico relativo, no similitud estética.

---

## 3. Qué significa y qué no significa que una predicción "se vea diferente"

### Lo que NO implica automáticamente:

- Que el modelo esté aprendiendo mal o esté mal entrenado.
- Que la métrica sea incorrecta o esté inflada artificialmente.
- Que el modelo sea menos útil que uno cuya máscara se vea más parecida visualmente.
- Que exista un error de implementación o de preprocesamiento.

### Lo que SÍ puede implicar, en función del contexto:

- Que el desbalance de clases está limitando la detección de `big_rock`, lo cual es un problema estructural del dataset, no del modelo.
- Que los bordes predichos son más suavizados que los anotados manualmente, lo cual es inherente a los modelos de segmentación supervisados.
- Que el modelo produce predicciones razonables en zonas que el anotador marcó como `ignore`, lo cual es invisible a las métricas pero visible a la inspección.

### Cuándo sí sería una señal de problema real:

- Si el modelo predice sistemáticamente **una sola clase** para toda la imagen (colapso de clases). Esto se detecta inspeccionando la distribución de predicciones por clase.
- Si las métricas por clase muestran IoU ≈ 0 para clases que deberían ser detectables (por ejemplo, bedrock con IoU < 0.5 cuando ocupa el 50% de los píxeles).
- Si la máscara predicha es completamente aleatoria o no coincide con ninguna estructura visual de la imagen de entrada.
- Si existe evidencia de fuga de datos entre train y test (violación del split fijo), lo que inflaría artificialmente las métricas.
- Si la varianza entre seeds es extremadamente alta (σ > 0.05 en mIoU), lo que sugeriría inestabilidad en el entrenamiento o en el protocolo de evaluación.

En el proyecto AI4MARS, ninguno de estos indicadores de problema real está presente en SegFormer-B2: el modelo detecta las cuatro clases, su varianza entre seeds es σ = 0.0063 (la más baja de todos los modelos), y las métricas por clase son consistentes con el grado de dificultad esperado según la distribución del dataset.

---

## 4. Interpretación de ejemplos visuales del proyecto

Al inspeccionar comparaciones imagen–GT–predicción en el notebook `05f_model_segformer.ipynb`, es frecuente observar los siguientes patrones, que se interpretan como sigue:

**Patrón 1 — Regiones de `big_rock` no detectadas o parciales.** El modelo predice bedrock o soil donde la GT indica big_rock. Explicación principal: `big_rock` representa solo el 0.89% de los píxeles; incluso con peso 3.58× en la pérdida, la señal de gradiente es escasa. Modelos sin mecanismo de atención global (campo receptivo limitado) tienen mayor dificultad aún. El IoU resultante (0.459 para SegFormer-B2) confirma que el modelo detecta casi la mitad del área real de big_rock, que es el mejor resultado comparable entre modelos CNN puros.

**Patrón 2 — Bordes ligeramente desplazados entre bedrock y soil.** Las fronteras predichas pueden diferir 5–15 píxeles de la GT, produciendo una diferencia visual notable aunque el IoU global permanezca alto. Esto es consecuencia directa de la ambigüedad de borde en imágenes de baja textura y la naturaleza continua de la pérdida.

**Patrón 3 — Zonas `ignore` (255) con predicción visible.** El modelo predice una clase en píxeles que la GT marca como 255. Visualmente produce una región coloreada donde la GT es neutra. Estas predicciones son completamente ignoradas por la loss y las métricas, por lo que no deben interpretarse como errores.

**Patrón 4 — Máscara predicha más "limpia" que la GT.** La GT puede contener regiones fragmentadas o ruidosas (resultado del proceso de crowdsourcing con múltiples anotadores). El modelo tiende a producir regiones más coherentes espacialmente, lo que puede verse "diferente" pero no necesariamente "peor".

---

## 5. Sección de interpretación para defensa oral

### Pregunta: "¿Por qué las predicciones no calcan la máscara de referencia?"

**Respuesta directa:** Porque la coincidencia visual exacta no es ni el objetivo ni el criterio de evaluación. El modelo se entrena para minimizar una función de pérdida definida sobre la distribución de píxeles observada, y se evalúa mediante métricas de solapamiento geométrico (IoU) que son independientes de la apariencia visual de la máscara. Una predicción puede diferir visualmente de la GT por cuatro razones no problemáticas: desbalance extremo de clases, ambigüedad inherente en bordes, ruido en las etiquetas de referencia, y píxeles marcados como `ignore` que son visibles en la imagen pero invisibles para la métrica.

**Cómo seguir:** "Lo que importa es si el modelo identifica correctamente el tipo de terreno donde existen etiquetas válidas. Para SegFormer-B2, el mIoU en el gold test es 0.8337 con desviación estándar de 0.0063 entre tres entrenamientos independientes. Las métricas por clase muestran que el modelo detecta correctamente bedrock con IoU de 0.940, sand con 0.958, y soil con 0.978. La clase más difícil, `big_rock`, alcanza IoU de 0.459, resultado consistente con su representación de menos del 1% de los píxeles en el dataset."

### Pregunta: "¿Cómo sabe que el resultado sigue siendo válido si visualmente se ve mal?"

**Respuesta:** El gold test set fue construido con 322 imágenes cuyas máscaras fueron validadas por científicos del JPL con acuerdo del 100% entre al menos 3 anotadores especializados. Las métricas se calculan exclusivamente sobre los píxeles con etiqueta válida (excluyendo ignore=255), usando el mismo split fijo para todos los modelos. La validez del resultado está garantizada por el protocolo experimental, no por la apariencia de la máscara. Adicionalmente, la baja varianza entre seeds (σ = 0.0063) indica que los resultados son reproducibles y no dependientes de la inicialización aleatoria.

### Cuándo sí sería una señal de problema real:

Si el jurado pregunta cómo se distingue una discrepancia "normal" de una problemática, la respuesta es: se diagnostica con métricas, no con inspección visual. Una predicción que colapsa en una sola clase para toda la imagen tiene IoU ≈ 0 en las clases ausentes. Un modelo que no generaliza tiene alta varianza entre seeds. Un dataset con fuga de datos produce un gap grande entre métricas de validación y test. Ninguno de estos indicadores está presente en los resultados del proyecto.
