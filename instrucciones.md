# Entregable de Deep Learning
## Estado del Arte, EDA Mejorado e Implementación de Modelos Benchmark

### Objetivo del Entregable
Identificar el estado del arte en Deep Learning, realizar un EDA exhaustivo e implementar un conjunto de modelos benchmark (Top 5 o superior) con buenas prácticas experimentales, garantizando reproducibilidad y evitando data leakage.

---

## 1. Definición del Problema y Dataset
* **Definición precisa** del problema.
* **Justificación** del dataset.
* **Descripción técnica**:
  * Tamaño y estructura.
  * Tipo de datos.
  * Variable objetivo.

---

## 2. Análisis Exploratorio de Datos (EDA) Mejorado
El EDA debe ser exhaustivo, profundo, reproducible y orientado a decisiones de modelado. No se aceptan análisis superficiales. Cada análisis debe responder explícitamente: **¿cómo impacta esto el diseño del modelo o el preprocesamiento?**

### EDA General
* **Análisis de la variable objetivo**:
  * Distribución (histogramas, densidades o conteos por clase).
  * Identificación de desbalance (ratio entre clases).
  * Detección de clases raras o poco representadas.
  * *Implicación:* Necesidad de técnicas como resampling, ponderación de pérdidas o métricas robustas.
* **Calidad del dataset**:
  * Valores faltantes (si aplica).
  * Datos corruptos o inconsistentes.
  * Duplicados (especialmente crítico en imágenes y texto).
* **Identificación de sesgos**:
  * Sesgos de representación (clases dominantes).
  * Sesgos temporales (en series de tiempo).
  * Sesgos semánticos (en NLP).
  * *Implicación:* Necesidad de balanceo, recolección adicional o validación estratificada.
* **Partición del dataset**:
  * Definir explícitamente conjuntos de entrenamiento, validación y prueba.
  * Justificar estrategia (aleatoria, estratificada, temporal).

### EDA Específico por Tipo de Dataset
#### 1. Imágenes
* **Inspección visual:** Visualizar muestras aleatorias por clase e identificar ruido, artefactos o errores de etiquetado.
* **Dimensionalidad y formato:** Distribución de tamaños (alto, ancho) y número de canales (RGB, escala de grises). *Implicación:* Necesidad de resizing, padding o normalización.
* **Distribución de intensidades:** Histogramas de píxeles por canal y detección de saturación o bajo contraste.
* **Balance de clases:** Conteo de imágenes por clase. *Implicación:* Data augmentation o class weighting.
* **Correlación espacial:** Identificación de patrones visuales relevantes. *Implicación:* Elección de arquitectura (CNN, Vision Transformers).

#### 2. Series de Tiempo
* **Visualización temporal:** Gráficos de la serie e identificación de tendencia, estacionalidad y ruido.
* **Estacionariedad:** Análisis visual y pruebas estadísticas (ADF si aplica). *Implicación:* Diferenciación o transformaciones.
* **Dependencia temporal:** ACF y PACF junto con la identificación de rezagos relevantes.
* **Análisis de outliers:** Detección de picos o anomalías y evaluación de su impacto en el modelo.
* **Frecuencia y granularidad:** Intervalo temporal y datos faltantes en el tiempo.
* **Implicaciones clave:** Selección de modelos (LSTM, GRU, Transformers), definición de ventanas temporales y evitar data leakage temporal.

#### 3. Procesamiento de Lenguaje Natural (NLP)
* **Longitud de textos:** Distribución de longitud de documentos. *Implicación:* Truncamiento o padding.
* **Análisis léxico:** Frecuencia de palabras e identificación de palabras comunes y raras.
* **Tokenización:** Evaluación del vocabulario y tamaño del diccionario.
* **Ruido en texto:** Caracteres especiales, URLs, stopwords. *Implicación:* Limpieza y normalización.
* **Balance de clases:** Distribución de etiquetas.
* **Implicaciones clave:** Selección de embeddings (Word2Vec, BERT, etc.) y elección de arquitectura (RNN, Transformers).

> ### Requisito obligatorio para el EDA
> Cada análisis del EDA debe concluir con una sección explícita de:
> * Decisión de preprocesamiento derivada.
> * Impacto en la arquitectura del modelo.
> * Riesgos identificados (incluyendo posibles fuentes de data leakage).
> * **No se aceptan análisis descriptivos sin interpretación técnica.**

---

## 3. Revisión del Estado del Arte
Se debe identificar un **Top 5 (o más)** modelos de Deep Learning relevantes para el problema, basado en literatura científica reciente (preferiblemente Q1 en WoS o Scopus).

### 3.1 Estrategia de Búsqueda (Obligatoria)
* Definir palabras clave utilizadas (ej. *"deep learning + forecasting + renewable energy"*).
* Bases de datos consultadas (WoS, Scopus, IEEE Xplore).
* Periodo de búsqueda (preferiblemente últimos 5-8 años).
* Criterios de inclusión/exclusión (relevancia al problema, uso de Deep Learning, métricas reportadas).

### 3.2 Identificación del Top de Modelos
Identificar al menos 5 modelos (idealmente 5-10), priorizando aquellos que sean: más citados, más recientes y con mejor desempeño reportado.

### 3.3 Análisis Detallado por Modelo
Para cada uno se debe incluir:
* Referencia completa del paper (formato académico).
* Tipo de arquitectura (CNN, RNN, LSTM, GRU, Transformer, híbridos, etc.).
* Descripción técnica (componentes principales, flujo de datos, innovación respecto a modelos previos).
* Tipo de problema que aborda y dataset(s) utilizados en el paper.
* Métricas reportadas y valores obtenidos.
* Fortalezas (capacidad de generalización, robustez, eficiencia) y limitaciones (sobreajuste, costo computacional, dependencia de datos).
* Complejidad computacional (número aproximado de parámetros y tiempo de entrenamiento si se reporta).

### 3.4 Comparación Crítica
* Comparación estructurada entre modelos (desempeño, robustez, escalabilidad).
* Identificación de tendencias (uso de Transformers vs modelos clásicos, modelos híbridos).
* Identificación de gap en la literatura.
* *Nota:* No se acepta una revisión descriptiva superficial. Se espera análisis crítico comparativo y justificación clara.

---

## 4. Implementación de Modelos Benchmark
Todos los modelos identificados deben ser implementados, comparados bajo el mismo entorno experimental y evaluados de forma rigurosa.

### 4.1 Diseño Experimental
* Definición clara de variables de entrada y variable objetivo.
* División del dataset (entrenamiento / validación / prueba) y justificación de la estrategia (aleatoria, estratificada o temporal).
* Definición de pipeline (preprocesamiento, entrenamiento, evaluación).

### 4.2 Implementación de Modelos
Para cada modelo se requiere:
* Implementación desde cero o uso de librerías justificadas.
* Definición de hiperparámetros (learning rate, número de capas, batch size, función de pérdida).
* Estrategia de entrenamiento (número de épocas, early stopping).
* Registro de curvas de entrenamiento y tiempo computacional.

### 4.3 Requisitos Experimentales Mejorados
* Control de semillas aleatorias.
* Repetición de experimentos (**mínimo 3 runs**).
* Reporte de **media y desviación estándar**.
* Uso de GPU (si aplica) debidamente documentado.

> ### Buenas Prácticas Obligatorias: Prevención de Data Leakage
> * Separar completamente train/validation/test.
> * Ajustar transformaciones solo en train.
> * Evitar mezcla temporal en series de tiempo.
> * Evitar fuga de información semántica en NLP.
> * Validación consistente entre modelos.

### 4.4 Evaluación
Definir métricas acordes al problema:
* **Clasificación:** Accuracy, Precision, Recall, F1-score, AUC-ROC.
* **Regresión:** RMSE, MAE, $R^{2}$.
* **Segmentación de imágenes:**
  * IoU (Intersection over Union):  
    $$IoU=\frac{TP}{TP+FP+FN}$$
  * Dice Coefficient (F1-score para segmentación):  
    $$Dice=\frac{2TP}{2TP+FP+FN}$$
  * Pixel Accuracy (proporción de píxeles correctamente clasificados).
  * Mean IoU (mIoU).
  * Boundary-based metrics (opcional).

#### Análisis de Resultados e Igualdad de Condiciones
* **Comparación justa:** Mismo dataset, misma partición, mismo pipeline de preprocesamiento y misma configuración de evaluación.
* **Análisis cuantitativo:** Reportar media y desviación estándar (múltiples ejecuciones), análisis de varianza entre modelos, evaluación de robustez e identificación de overfitting/underfitting.
* **Análisis cualitativo (especialmente en segmentación):** Visualización de predicciones vs ground truth, identificación de errores sistemáticos y desempeño en casos difíciles.

---

## 5. Paper Guía
Seleccionar un paper base que sirva como referencia principal del proyecto:
* Referencia completa (formato académico).
* Descripción detallada del modelo original (arquitectura y supuestos).
* Justificación de selección (relevancia e impacto en la literatura).
* Relación con los modelos benchmark (similitudes y diferencias).

---

## 6. Benchmark Final
El benchmark debe ser reproducible, justo y comparable.

### 6.1 Comparación Tabular
Construir una tabla con:
* Modelo.
* Métricas (media $\pm$ desviación estándar).
* Intervalos de confianza (95%).
* Tiempo de entrenamiento.
* Número de parámetros (recomendado).

### 6.2 Ranking de Modelos
* Ordenar modelos según la métrica principal del problema.
* Analizar trade-off: Precisión vs costo computacional y Precisión vs complejidad del modelo.

### 6.3 Análisis Estadístico y Pruebas de Hipótesis
Se deben realizar pruebas de hipótesis formales para evaluar si las diferencias entre modelos son estadísticamente significativas. **No se aceptan comparaciones basadas únicamente en promedios.**

#### Pruebas según el escenario:
* **Comparación General entre Modelos:**
  * Comparaciones pareadas: *t-test pareado* (con normalidad) o *Wilcoxon signed-rank test* (no paramétrico).
  * Múltiples modelos: *ANOVA* (si aplica) o *Friedman test* (recomendado).
  * Post-hoc: *Nemenyi test* o *Corrección de Bonferroni o Holm*.
* **Caso Específico - Clasificación (AUC):**
  * *Test de DeLong* para comparación estadística entre curvas ROC y evaluación de diferencias significativas en AUC (recomendado sobre el mismo dataset). Reportar diferencia de AUC, p-valor e intervalos de confianza.
* **Caso Específico - Series de Tiempo:**
  * *Test de Diebold-Mariano* (precisión predictiva).
  * *Test BDS* (independencia y no linealidad en residuos).
  * *Test de Ljung-Box* (autocorrelación en residuos).
* **Caso Específico - Regresión:**
  * *t-test o Wilcoxon* sobre errores (RMSE, MAE) y evaluación de intervalos de confianza de métricas.

#### Buenas Prácticas Estadísticas
* Nivel de significancia: $\alpha = 0.05$.
* Reportar p-valores exactos (no solo $< 0.05$).
* Reportar intervalos de confianza y verificar supuestos (normalidad, independencia).
* Realizar múltiples ejecuciones (mínimo 3–5).
* *Interpretación:* Distinguir entre significancia estadística y relevancia práctica, evitando conclusiones basadas solo en métricas promedio.

### 6.4 Discusión Crítica
Responder con base en la evidencia estadística rigurosa:
* ¿Qué modelo funciona mejor y por qué?
* ¿Qué modelo generaliza mejor?
* ¿Las diferencias son estadísticamente significativas?
* ¿Existe trade-off relevante entre desempeño y costo?
* Limitaciones del estudio y recomendaciones futuras.

---

## Entregables Finales
1. Informe en **Jupyter Book**.
2. **Código completo** (EDA + modelos + análisis estadístico).
3. **Dataset**.
4. **Tabla comparativa** con métricas e inferencia estadística.

## Evaluación
| Criterio | Peso |
| :--- | :--- |
| EDA Mejorado | 25 % |
| Revisión de literatura | 25 % |
| Implementación modelos | 30 % |
| Benchmark y análisis estadístico | 20 % |

---
**Observación Final:** Se espera un análisis exhaustivo: no solo comparar modelos, sino demostrar rigurosamente, mediante inferencia estadística, la existencia (o no) de diferencias significativas.