# 04 — Revisión del Estado del Arte

## 4.1 Estrategia de Búsqueda

Se realizaron búsquedas exhaustivas en bases de datos académicas Q1 (Scopus, Science Direct, Web of Science, IEEE Xplore, arXiv) durante los últimos 8 años (2018–2026), priorizando publicaciones con alto impacto en segmentación semántica para navegación de rovers marcianos.

Las búsquedas se construyeron combinando términos de dominio y datasets mediante operadores booleanos (AND, OR). Los **términos de dominio** empleados fueron: `semantic segmentation`, `terrain classification`, `Mars rover navigation`, `planetary surface analysis`, `autonomous path planning`. Los **datasets y benchmarks** de referencia fueron: `AI4MARS`, `Mars-Seg`, `S5Mars`, `NavCam`, `HazCam`.

Criterios de **inclusión**: uso explícito de Deep Learning (CNN, Transformers), evaluación en AI4MARS o datasets similares (Mars-Seg, S⁵mars), métricas reportadas (mIoU, IoU, Accuracy), reproducibilidad, publicación peer-reviewed o preprint verificable. Criterios de **exclusión**: enfoques no-DL, datasets terrestres sin adaptación marciana, publicaciones pre-2018 o sin métricas cuantitativas, trabajos sin publicación académica verificada.

---

## 4.2 Identificación de la Propuesta Original y los Benchmarks

Se identificó una propuesta original del proyecto (**SegFormer-B2 adaptado a AI4MARS**) y cinco modelos benchmark relevantes con publicación académica verificada o arquitectura reciente reproducible. La selección prioriza rendimiento reportado en AI4MARS o datasets marcianos equivalentes, recencia (2021–2025) y cobertura representativa del espectro de arquitecturas CNN–Transformer–Híbrido.

1. **DeepLabV3+ (Mohammad et al., 2024)**: CNN encoder-decoder con atrous convolutions y augmentación GAN (SemanticStyleGAN). Evaluado en AI4MARS: mIoU 87–88%, Acc 99%. Mejor modelo en AI4MARS estándar; el GAN augmentation aporta +2% en clases raras.

2. **MarsSeg (Li et al., 2024)**: Encoder-decoder con Feature Enhancement Layer (Mini-ASPP + PSA + SPPM) y Focal-Dice loss adaptada al desbalance marciano. Evaluado en Mars-Seg MSL: mIoU 65.69%. Mejor IoU de Big Rock en AI4MARS: **80.89** (métrica por clase, no mIoU global).

3. **TerSeg (Fan et al., 2025)**: Dual-branch CNN (ResNet-34) + Swin Transformer con fusión ponderada FL/FG y módulo FLGA multi-escala. Evaluado en S⁵mars (9 clases): mIoU 71.96%, 22M parámetros, 26G FLOPs. Único sistema que integra segmentación + planificación de rutas (algoritmo TA*).

4. **DepthFormer (Ma et al., 2025)**: Swin Transformer con entrada 4-banda (RGB + profundidad estéreo) y Pyramid Pooling Module. Evaluado en DepthMars (dataset propio, rover Zhurong): mIoU 75.99%, aAcc 98.28%. Primera integración de profundidad para segmentación marciana.

5. **IC-TransUNet (Zhu et al., 2025)**: arquitectura CNN–Transformer dual-branch basada en InceptionNeXt y CSWin Transformer, adaptada en este proyecto como benchmark adicional para evaluar si una fusion local-global mas compleja supera a la propuesta SegFormer en AI4MARS.

La propuesta **SegFormer-B2** cubre el caso Transformer jerárquico eficiente. Los benchmarks cubren CNN especializado (DeepLabV3+, MarsSeg), Transformer con encoder Swin (DepthFormer), modelos híbridos CNN+Transformer (TerSeg) y una variante dual-branch reciente (IC-TransUNet). Esta cobertura permite evaluar si la mayor complejidad arquitectónica se traduce realmente en mejores resultados bajo el mismo protocolo AI4MARS.

---

## 4.3 Análisis Detallado por Modelo

### 4.3.1 DeepLabV3+ para Clasificación de Terreno Marciano

**Referencia completa:**
> Mohammad, F., Gao, Y., Kay, S., Field, R., De Benedetti, M., & Ntagiou, E. V. (2024). *Deep Learning based Semantic Segmentation for Mars Rover Terrain Classification*. 2024 International Conference on Space Robotics (iSpaRo), June 24–27, 2024, Luxembourg. IEEE. DOI: 10.1109/iSpaRo60631.2024.10687827

**Tipo de arquitectura:** CNN encoder-decoder (DeepLabV3+) con GAN de aumento de datos (SemanticStyleGAN).

**Descripción técnica:**

DeepLabV3+ es el modelo más reciente de la familia DeepLab (DeepLabV1 → V2 → V3 → V3+). Adopta una arquitectura encoder-decoder donde el encoder usa convoluciones atrous/dilatadas con diferentes *dilation rates*, efectivamente expandiendo el campo receptivo sin aumentar el número de parámetros. El decoder procesa las características mediante una cabeza de 1×1 Conv para reducir canales, seguido de upsampling bilinear ×4 y concatenación con características de bajo nivel del encoder, finalizando con una convolución 3×3 y upsampling ×4 final.

En este trabajo se evalúan dos backbones: **ResNet-50** y **ResNet-101** (preentrenados en ImageNet), y dos resoluciones de entrada: 256×256 y 512×512. La contribución principal es la integración de **SemanticStyleGAN** para generar imágenes sintéticas condicionadas en máscaras semánticas de las clases raras (Big Rock / Float Rock), duplicando el conjunto de entrenamiento para esas clases. Se excluye la clase background (ignore=255) del entrenamiento y evaluación.

- **Componentes principales:** Encoder DCNN con atrous convolutions (rates: 6, 12, 18), ASPP (Atrous Spatial Pyramid Pooling), Image Pooling, decoder con skip connections de bajo nivel.
- **Flujo de datos:** Imagen entrada → Encoder ResNet → ASPP module → 1×1 Conv → Upsampling ×4 → Concat con low-level features → 3×3 Conv → Upsampling ×4 → Máscara de salida.
- **Innovación:** Integración de GAN (SemanticStyleGAN + GauGAN) para data augmentation específica de clases raras en el dominio marciano. También se investiga el uso de early stopping para regularización.
- **Función de pérdida:** Cross-entropy (CE). Optimizador: SGD con polynomial LR decay (lr=0.001). Batch sizes optimizados según memoria GPU (RTX 4000, 8GB).

**Tipo de problema:** Segmentación semántica multi-clase (4 clases: Soil, Bedrock, Sand, Big Rock).

**Datasets utilizados:** AI4MARS (35,000 imágenes, splits 90/10 para train/val+test) y LabelMars (ESA, 5,000 imágenes, splits 80/10/10). En AI4MARS se usan los 3 test sets de referencia: M1, M2, M3 (255 imágenes c/u, validadas por especialistas).

**Métricas reportadas (AI4MARS, mejor configuración):**

| Configuración | Test | Acc | mIoU | IoU Soil | IoU Bedrock | IoU Sand | IoU Big Rock |
|---|---|---|---|---|---|---|---|
| DeepLabV3+ ResNet101 (512, 4) | M3 | 0.99 | **0.87** | 0.99 | 0.94 | 0.97 | 0.59 |
| DeepLabV3+ ResNet101 + GAN (512, 4) | M3 | 0.99 | **0.88** | 0.99 | 0.95 | 0.97 | **0.61** |
| U-Net ResNet101 (512, 4) — Baseline | M3 | 0.99 | 0.84 | 0.98 | 0.95 | 0.97 | 0.44 |

**Fortalezas:** Mejor modelo en AI4MARS del paper (+5% mIoU vs. U-Net en M3); el GAN augmentation mejora hasta 2% en clases raras; el early stopping reduce drásticamente el tiempo de entrenamiento (promedio: 26 epochs en AI4MARS); ResNet-50 como backbone es preferible para deployment (igual rendimiento que ResNet-101 con menos parámetros y mayor velocidad).

**Limitaciones:** La clase Big Rock sigue siendo la más difícil (IoU ≤0.61 incluso con GAN); el rendimiento cae significativamente en M1 y M2 respecto a M3, indicando dependencia del nivel de consenso en etiquetado; el aumento sintético con GAN añade complejidad al pipeline de entrenamiento.

**Complejidad computacional:** Backbone ResNet-50: ~25M params; ResNet-101: ~45M params. No se reporta FLOP count explícito. Hardware de entrenamiento: Quadro RTX 4000 (8GB); convergencia promedio: 26 epochs con early stopping.

---

### 4.3.2 MarsSeg: Segmentación de Superficie Marciana con Extractor y Conector Multi-nivel

**Referencia completa:**
> Li, J., Chen, K., Tian, G., Li, L., & Shi, Z. (2024). *MarsSeg: Mars Surface Semantic Segmentation with Multi-level Extractor and Connector*. arXiv preprint arXiv:2404.04155v1. Beihang University.

**Tipo de arquitectura:** CNN encoder-decoder con módulos de mejora de características en múltiples escalas (Mini-ASPP, PSA, SPPM).

**Descripción técnica:**

MarsSeg es una red encoder-decoder diseñada específicamente para las características del terreno marciano: variaciones de escala extremas, diferencias mínimas inter-clase (el terreno marciano carece de fauna, flora y agua, con escasa diversidad visual), y severo desbalance de clases (Big Rock ~2% en AI4MARS). La arquitectura reduce el número de capas de downsampling respecto a arquitecturas estándar para preservar detalles locales de grano fino.

La innovación central es la **capa de conexión de mejora de características** (Feature Enhancement Connection Layer) situada entre el encoder y el decoder, que funciona como un puente multi-escala compuesto por dos tipos de conectores:
- **Shadow Feature Connector**: apila módulos Mini-ASPP y PSA para capturar detalles locales y características de sombra/textura a escalas pequeñas.
- **Deep Feature Connector** (SPPM): captura características semánticas de alto nivel relacionadas con la categoría del terreno.

Los tres módulos clave:
1. **Mini-ASPP (Mini Atrous Spatial Pyramid Pooling):** Versión compacta del ASPP estándar. Aplica convoluciones dilatadas en paralelo con múltiples tasas para capturar contexto multi-escala sin overhead computacional excesivo. Orientado a expresar detalles locales y objetos pequeños.
2. **PSA (Polarized Self-Attention):** Módulo de auto-atención polarizada que opera en canales y dimensiones espaciales. Usa atención tipo softmax para la dimensión canal y sigmoid para la dimensión espacial, adaptado para la extracción de características de sombra en terreno marciano.
3. **SPPM (Strip Pyramid Pooling Module):** Pooling piramidal en forma de franja para captura de contexto global a nivel semántico profundo, extrayendo características relacionadas con la categoría de terreno a alta escala.

**Función de pérdida:** Focal-Dice combinada con pesos adaptativos, diseñada para el desbalance de clases marciano. **Optimizador:** SGD (lr=0.001, momentum=0.9, weight decay=0.0001). **Augmentación:** Random crop, flip y random scaling para Mars-Seg; augmentación enfocada en Big Rock para AI4MARS.

**Flujo de datos:** Imagen → Encoder (CNN con downsampling reducido) → Feature Enhancement Layer (Mini-ASPP + PSA + SPPM) → Decoder → Máscara de segmentación.

**Tipo de problema:** Segmentación semántica multi-clase (4 clases en AI4MARS; 9 clases en Mars-Seg).

**Datasets utilizados:** Mars-Seg (MSL-Seg: 4,155 imágenes RGB 560×500; MER-Seg: 1,024 imágenes grayscale 1024×1024), split 8:2. AI4MARS MSL subset (16,064 entrenamiento, 322 test).

**Métricas reportadas:**

En **Mars-Seg (MSL-Seg)**, frente a DeepLabV3+ y SegFormer:

| Método | mIoU |
|---|---|
| DeepLabV3+ | 52.33 |
| SegFormer | 58.07 |
| **MarsSeg (Ours)** | **65.69** |

En **AI4MARS**, MarsSeg alcanza un **IoU de Big Rock de 80.89** — el más alto reportado en este subset entre los modelos comparados — con un mIoU global de 80.89 sobre las 4 clases. Esta métrica por clase es relevante dado que Big Rock (~2% de píxeles) es el cuello de botella universal en la segmentación marciana.

**Fortalezas:** Mejor generalización inter-dataset (Mars-Seg y AI4MARS) con una sola arquitectura; el SPPM mejora significativamente las clases de alta semántica; la combinación focal-dice loss resuelve el desbalance severo; rendimiento superior a SegFormer en escenas con baja distinguibilidad inter-clase (imágenes en escala de grises del MER-Seg: mIoU 52.48 vs. SegFormer 50.36).

**Limitaciones:** Rendimiento en Big Rock en grayscale (MER-Seg) permanece bajo (IoU 26.30) dada la ausencia de información de color; las variaciones de escala extremas entre MSL y MER complican la generalización; el número exacto de parámetros no se reporta explícitamente en el paper.

**Complejidad computacional:** No reportado explícitamente. Implementación en PyTorch. Entrenamiento en NVIDIA RTX 3090 (24GB), batch 8.

---

### 4.3.3 TerSeg: Red de Segmentación Semántica Dual-Branch para Terreno Marciano y Planificación de Rutas

**Referencia completa:**
> Fan, L., Yuan, J., & Zha, K. (2025). *TerSeg: A dual-branch semantic segmentation network for Mars terrain and autonomous path planning*. Expert Systems with Applications, 270, 126397. Elsevier. DOI: 10.1016/j.eswa.2025.126397. (Received 7 Feb 2024; Accepted 2 Jan 2025; Available online 15 Jan 2025).

**Tipo de arquitectura:** Híbrida CNN + Vision Transformer (dual-branch), con módulos FL, FG y FLGA para fusión multi-escala de características locales y globales.

**Descripción técnica:**

TerSeg propone una arquitectura de red de fusión de dos ramas (*dual-branch fusion network*) que combina explícitamente las fortalezas complementarias de CNNs (captura de características locales, texturas, bordes) y Transformers (captura de contexto global de largo alcance):

- **Branch_T (Transformer Branch):** Utiliza **Swin Transformer** como backbone para extracción de características globales. Consta de 4 etapas con 2, 2, 6 y 2 módulos SwinTB respectivamente. Las características `T_n = {t₁, t₂, t₃, t₄}`.
- **Branch_C (CNN Branch):** Utiliza **ResNet-34** como backbone para extracción de características locales. Compuesto por 1 bloque convolucional inicial y 4 Residual Blocks (RBs). Las características `R_n = {r₁, r₂, r₃, r₄}`.

**Módulos de fusión:**
- **FL (Focus on Local):** Fusiona características de bajo nivel donde la rama CNN domina (estadíos tempranos). Aplica combinación ponderada priorizando r_i sobre t_i.
- **FG (Focus on Global):** Fusiona características en estadíos tardíos donde el Transformer domina. Prioriza t_i sobre r_i para captura semántica global.
- **FLGA (Focus on Local and Global Feature Aggregation):** Módulo de agregación final que funde las feature maps de múltiples escalas m_s = {m₁, m₂, m₃, m₄} para restaurar la resolución de la máscara de predicción.

La configuración óptima encontrada es **TC_S** con ~**22M parámetros** y **26G FLOPs**.

Adicionalmente, el paper propone el algoritmo **TA\*** (Terrain-Aware A*) que utiliza la máscara de segmentación de TerSeg, lógica difusa para evaluar la atravesabilidad de cada clase de terreno (Soil: High, Sand: Low, Bedrock/Rock: Poor), y mapas de peligro/seguridad para planificación de rutas autónoma.

**Función de pérdida:** Focal Loss (α=0.25, γ=2.0). **Optimizador:** Adam con ReduceLROnPlateau.

**Tipo de problema:** Segmentación semántica de terreno espacial para path planning autónomo.

**Datasets utilizados:** S⁵mars (9 categorías: Sky, Ridge, Soil, Sand, Bedrock, Rock, Rover, Trace, Hole) y AI4MARS (4 categorías). Los resultados principales del paper se reportan en **S⁵mars**.

**Métricas reportadas (S⁵mars, comparación vs. estado del arte):**

| Método | PA(%) | Precision(%) | MIoU(%) | FWIoU(%) | Fscore(%) |
|---|---|---|---|---|---|
| DeepLabV3+ | 87.66 | 78.78 | 70.50 | 79.04 | 81.17 |
| UNet | 86.40 | 74.90 | 66.27 | 76.93 | 79.31 |
| SegFormer | 84.83 | 58.15 | 50.23 | 75.04 | 59.74 |
| CGGLNet | 90.96 | 71.35 | 66.00 | 84.25 | 72.91 |
| **TerSeg (Ours)** | **89.90** | **80.48** | **71.96** | **82.33** | **81.63** |

Recall por clase en S⁵mars: TerSeg logra **99.83%** en Ridge y **89.02%** en Rock, superando a todos los métodos comparados en precisión y recall balanceado.

**Fortalezas:** Eficiencia computacional destacada (22M params, 26G FLOPs) con rendimiento superior al de arquitecturas más pesadas; el balance local-global mejora ~3% mIoU vs. DeepLabV3+ en S⁵mars; integración única con TA* como sistema end-to-end de segmentación y planificación de rutas; robusto en clases de baja frecuencia (Trace, Hole).

**Limitaciones:** Los resultados principales son en S⁵mars (9 clases), no directamente comparables en mIoU con los benchmarks AI4MARS estándar (4 clases); SegFormer en este dataset rinde significativamente por debajo (MIoU 50.23%), sugiriendo que los Transformers puros sufren con la complejidad multi-clase en escenas marcianas; el método TA* requiere post-procesamiento adicional que añade latencia.

**Complejidad computacional:** ~22M parámetros, ~26G FLOPs (configuración TC_S). Hardware: GeForce RTX 3080 Ti (12GB).

---

### 4.3.4 DepthFormer: Red Transformer Depth-Enhanced para Segmentación Semántica de la Superficie Marciana

**Referencia completa:**
> Ma, Y., Li, Z., Wu, B., & Duan, R. (2025). *DepthFormer: Depth-Enhanced Transformer Network for Semantic Segmentation of the Martian Surface From Rover Images*. Earth and Space Science, 12, e2024EA003812. AGU/Wiley. DOI: 10.1029/2024EA003812. (Received: 15 Oct 2024; Accepted: 25 May 2025).

**Tipo de arquitectura:** Transformer encoder (Swin Transformer) con entrada multimodal 4-banda (RGB + profundidad), Pyramid Pooling Module (PPM) como cuello de botella, y decoder multi-escala con fusión por suma elemento a elemento.

**Descripción técnica:**

DepthFormer es la primera arquitectura de deep learning que incorpora información de **profundidad estimada** como cuarta banda de entrada para la segmentación semántica del terreno marciano. La profundidad se obtiene a partir de imágenes estéreo del rover Zhurong mediante fotogrametría (pipeline SfM + MVS). La idea central es que la profundidad resuelve ambigüedades de textura y apariencia comunes en el terreno marciano (por ejemplo, distinguir cráteres de suelo liso).

**Arquitectura detallada:**
1. **Patch Partition:** El tensor de entrada 2048×2048×4 (RGB + profundidad) se particiona en patches → tensor 512×512×64.
2. **Encoding (Swin Transformer, 4 etapas):** Etapa 1: 512×512×96 (×2 SwinTB); Etapa 2: 256×256×192 (×2 SwinTB); Etapa 3: 128×128×384 (×6 SwinTB); Etapa 4: 64×64×768 (×2 SwinTB). Cada bloque usa W-MSA y SW-MSA alternados para capturar dependencias locales y de largo alcance con coste computacional reducido.
3. **PPM (Pyramid Pooling Module):** Cuello de botella entre encoder y decoder. Captura contexto a múltiples escalas mediante pooling paralelo con diferentes tamaños de ventana.
4. **Decoding:** Feature maps fusionadas mediante sumas elemento a elemento con upsample bilinear sucesivo. Convoluciones 1×1 y 3×3 para alineación de canales. Decode Head final.

**Dataset propio (DepthMars):** 623 imágenes del rover Zhurong (2048×2048 px), con 403 train / 110 val / 110 test. 5 clases: Soil (45.01%), Rock (1.23%), Sand (12.67%), Craters (0.01%), Others (41.97%).

**Función de pérdida:** Weighted Cross-Entropy (WCE) con pesos manuales: Others=0.1, Soil=0.3, Rock=8, Sand=3, Craters=5. **Optimizador:** AdamW con LR policy "poly" (lr inicial = 1×10⁻⁶). **Entrenamiento:** NVIDIA RTX 2080Ti, batch 16, 160K iteraciones, crop 512×512.

**Métricas reportadas (DepthMars dataset):**

| Método | Soil IoU | Rock IoU | Sand IoU | Craters IoU | Others IoU | aAcc | mIoU |
|---|---|---|---|---|---|---|---|
| DeepLabV3 | 74.81 | **99.00** | 69.20 | 17.85 | **98.82** | 95.17 | 71.94 |
| SegFormer | 96.31 | 42.96 | 93.26 | 3.86 | 98.55 | 97.91 | 66.99 |
| Mask2Former | 96.17 | 57.21 | 92.46 | 34.75 | 98.59 | 97.87 | 75.84 |
| Swin Transformer | 92.67 | 54.09 | 91.47 | 34.52 | 95.14 | 95.85 | 73.58 |
| **DepthFormer** | 93.82 | 60.25 | **92.66** | **38.45** | 94.78 | **98.28** | **75.99** |

El uso de WCE mejora mIoU de 71.31 a **75.99** (sin WCE → con WCE), con mejora de +20 pts IoU en Rock a expensas de una ligera reducción en Soil y Others.

**Fortalezas:** Primera integración de profundidad estéreo para segmentación semántica marciana; la cuarta banda aporta información de relieve que resuelve ambigüedades de apariencia; WCE mejora sustancialmente clases minoritarias; Swin Transformer ofrece escalabilidad con balance localidad-globalidad; mayor aAcc entre todos los métodos comparados (98.28%).

**Limitaciones:** El dataset DepthMars (623 imágenes del rover Zhurong) es significativamente más pequeño que AI4MARS, limitando la comparabilidad directa; la generación del mapa de profundidad requiere imágenes estéreo (disponibles en Zhurong pero no universalmente en rovers NASA); los cráteres (0.01% del dataset) siguen siendo la clase más difícil (IoU 38.45%); benchmark no replicado por terceros.

**Complejidad computacional:** Swin Transformer preentrenado (ade20k, tiny). ~28M parámetros estimados. Hardware: NVIDIA RTX 2080Ti (11GB), batch 16.

---

### 4.3.5 SegFormer-B2 como Modelo Original del Proyecto

**Referencia de la arquitectura:**
> Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. Advances in Neural Information Processing Systems (NeurIPS), 34, 12077–12090.

**Justificación como propuesta original:**
SegFormer fue originalmente propuesto para segmentación semántica general. En este proyecto se usa como propuesta original aplicada al dominio AI4MARS: una arquitectura Transformer jerárquica, eficiente y no especializada originalmente en Marte, evaluada bajo un protocolo común con gold test fijo y múltiples seeds. Su papel no es ser un benchmark más, sino funcionar como el modelo central que se contrasta contra CNNs, Transformers Swin e híbridos CNN+Transformer.

**Tipo de arquitectura:** Transformer encoder jerárquico (Mix Transformer, MiT) + decoder MLP lightweight (All-MLP Decoder).

**Descripción técnica:**

SegFormer combina un encoder Transformer jerárquico con un decoder MLP ultra-ligero, lo que lo hace significativamente más eficiente que arquitecturas basadas en ViT puro. Su diseño evita la interpolación posicional, haciéndolo robusto a cambios de resolución en inferencia.

**Encoder MiT (Mix Transformer):**
- Encoder jerárquico con 4 etapas de extracción de features multi-resolución.
- Cada etapa aplica *Overlapped Patch Merging* para mayor continuidad espacial.
- Bloques con **Efficient Self-Attention** (reducción de la secuencia K/V mediante *Sequence Reduction* con ratio R∈{64,16,4,1} por etapa) para reducir la complejidad de O(N²) a O(N²/R).
- **Mix-FFN** (Feed-Forward Network con 3×3 depthwise conv) que combina información local y posicional, eliminando la necesidad de positional encoding explícito.
- Variante usada en este proyecto: **MiT-B2** (~25M params).

**Decoder All-MLP:**
- Recibe feature maps de las 4 etapas del encoder.
- Aplica una capa MLP individual a cada escala para unificar dimensiones de canal.
- Concatena y aplica una MLP de fusión + Conv 1×1 para la predicción final.
- Computacionalmente trivial comparado con el ASPP de DeepLabV3+.

**Función de pérdida:** Cross-entropy estándar con `ignore_index=255`. Preentrenamiento: ImageNet-1K (MiT-B2). Fine-tuning en AI4MARS.

**Tipo de problema:** Segmentación semántica de terreno (4 clases: Soil, Bedrock, Sand, Big Rock).

**Métricas reportadas en dominio marciano:**

| Dataset | mIoU | Acc | Fuente |
|---|---|---|---|
| AI4MARS M3 (gold set) | 83.55% | 90.86% | Mohammad et al. (2024); Fan et al. (2025) |
| Mars-Seg MSL-Seg | 58.07% | — | Li et al. / MarsSeg (2024) |
| S⁵mars | 50.23% | 84.83% | Fan et al. / TerSeg (2025) |
| DepthMars | 66.99% | 97.91% | Ma et al. / DepthFormer (2025) |

**Fortalezas:** Encoder jerárquico multi-escala sin positional encoding → robusto a variación de resolución en imágenes de rovers; decoder MLP extremadamente ligero; buen rendimiento en AI4MARS estándar (83.55%); diseño modular (B0–B5) permite ajuste de capacidad según hardware.

**Limitaciones:** En datasets con mayor complejidad (9 clases en S⁵mars) el rendimiento cae dramáticamente (50.23% mIoU); la atención global sin inductive bias local sufre con texturas sutiles y clases poco representadas en escenas marcianas complejas; en el paper de TerSeg es el modelo con menor rendimiento en S⁵mars entre los comparados.

**Complejidad computacional:** MiT-B2: ~25M parámetros, ~62G FLOPs. Decoder MLP: <1M params adicionales. Total SegFormer-B2: ~25M params.

---

### 4.3.6 IC-TransUNet como Benchmark CNN-Transformer Adicional

**Referencia de la arquitectura:**
> Zhu et al. (2025). *IC-TransUNet: CNN-transformer dual branch collaborative model for semantic segmentation of high-resolution remote sensing images*.

**Tipo de arquitectura:** modelo dual-branch CNN + Transformer, con rama convolucional InceptionNeXt, rama Transformer tipo CSWin, modulos de fusion colaborativa y cabeza MLP de segmentacion.

**Justificacion de inclusion como benchmark:**
IC-TransUNet se incorpora como benchmark adicional para probar una hipotesis concreta: si una arquitectura dual-branch mas compleja, diseñada para combinar textura local y contexto global, supera a SegFormer-B2 en imagenes marcianas de baja textura y clases desbalanceadas. Su inclusion es util porque representa una familia reciente de modelos hibridos de alta capacidad, distinta de TerSeg y de los Transformers puros.

**Adaptacion AI4MARS en este proyecto:**
El notebook `05b_model_transunet.ipynb` adapta la arquitectura a entrada RGB 256x256, cuatro clases validas (`soil`, `bedrock`, `sand`, `big_rock`) y `ignore_index=255`. El entrenamiento usa el split fijo MSL 6k, normalizacion calculada solo con train y evaluacion final sobre el gold test MSL min3. Para compensar el desbalance, combina CE, Dice y Focal Loss con pesos por clase y muestreo ponderado hacia imagenes con `big_rock`.

**Resultado esperado dentro del benchmark:**
IC-TransUNet se evalua con las mismas seeds `[42, 123, 7]` y checkpoints de Fase 1. En la narrativa del proyecto funciona como contraste: si no supera a SegFormer-B2, refuerza que una arquitectura mas compleja no necesariamente mejora el balance precision-estabilidad-costo en AI4MARS.

**Limitaciones:** La implementacion local depende de PyTorch y checkpoints generados en Colab Pro. El recalculo de metricas desde `.pth` debe ejecutarse en Colab o en un entorno local con `torch`; el Jupyter Book se construye con ejecucion desactivada y conserva los outputs/resultados agregados.

---

## 4.4 Comparación Crítica

### 4.4.1 Tabla Comparativa de Desempeño

La comparación directa está parcialmente limitada por el uso de distintos datasets en cada paper. Se consolida a continuación con las métricas disponibles en los datasets compartidos:

**Sobre AI4MARS (dataset del proyecto):**

| Modelo | mIoU (M3) | Acc | IoU Big Rock | Parámetros | Año |
|---|---|---|---|---|---|
| **DeepLabV3+ + GAN** | **88%** | **99%** | 61% | ~45M (R101) | 2024 |
| DeepLabV3+ (sin GAN) | 87% | 99% | 59% | ~25–45M | 2024 |
| SegFormer-B2 (propuesta del proyecto) | 83.55% | 90.86% | — | ~25M | 2021/2024 |
| MarsSeg | 80.89%* | — | **80.89%*** | N/R | 2024 |
| U-Net ResNet101 | 84% | 99% | 44% | ~45M | 2024 |

*MarsSeg reporta 80.89 como mIoU global sobre las 4 clases de AI4MARS; este valor coincide numéricamente con el IoU de Big Rock reportado en su tabla — debe interpretarse como mIoU del subset AI4MARS (4 clases), no comparable directamente con el M3 gold set de Mohammad et al.

**Sobre Mars-Seg (MSL-Seg):**

| Modelo | mIoU |
|---|---|
| **MarsSeg** | **65.69** |
| SegFormer | 58.07 |
| DeepLabV3+ | 52.33 |

**Sobre S⁵mars (9 clases):**

| Modelo | MIoU (%) | Params | FLOPs |
|---|---|---|---|
| **TerSeg** | **71.96** | **22M** | **26G** |
| DeepLabV3+ | 70.50 | ~25–45M | ~177G |
| UNet | 66.27 | ~31M | ~55G |
| CGGLNet | 66.00 | >45M | Alto |
| SegFormer | 50.23 | ~25M | ~62G |

**Sobre DepthMars (5 clases, rover Zhurong):**

| Modelo | mIoU | aAcc |
|---|---|---|
| **DepthFormer** | **75.99** | **98.28** |
| Mask2Former | 75.84 | 97.87 |
| Swin Transformer | 73.58 | 95.85 |
| DeepLabV3 | 71.94 | 95.17 |
| SegFormer | 66.99 | 97.91 |

**Sobre el benchmark unificado de este proyecto (AI4MARS MSL gold test):**

La comparacion final del proyecto se reporta en el notebook `06_benchmark_estadistico.ipynb`. Alli SegFormer-B2 se analiza como propuesta original y se contrasta contra DeepLabV3+, MarsSeg, TerSeg, DepthFormer-RGB e IC-TransUNet bajo el mismo split, las mismas seeds y las mismas metricas.

### 4.4.2 Tendencias Identificadas

**CNN clásicos vs. Transformers vs. Híbridos:**
- Los CNNs especializados (DeepLabV3+) dominan en **AI4MARS estándar** gracias a sus atrous convolutions y data augmentation con GAN, alcanzando hasta 88% mIoU.
- Los Transformers jerarquicos eficientes (SegFormer-B2) ofrecen una frontera atractiva de rendimiento, estabilidad y costo en AI4MARS; su degradacion en datasets multi-clase mas complejos reportados por la literatura (50.23% en S⁵mars de 9 clases) delimita el tipo de escenario donde los hibridos pueden ser preferibles.
- Los modelos **híbridos CNN+Transformer** (TerSeg, DepthFormer) representan el estado del arte más robusto: TerSeg logra el mejor balance eficiencia/rendimiento (22M params, 71.96% mIoU en S⁵mars), mientras DepthFormer introduce la modalidad de profundidad como ventaja diferencial en terrenos con relieve.
- **MarsSeg** destaca como el mejor modelo para la clase crítica Big Rock en AI4MARS, gracias a su Focal-Dice loss y arquitectura orientada a objetos pequeños.

**Modelos híbridos (CNN + Transformer):**
La tendencia clara de 2024–2025 es la arquitectura dual-branch donde Branch_C (CNN) captura detalles de textura local y Branch_T (Transformer) captura contexto semántico global. Esto supera tanto a los CNNs puros (sin contexto global suficiente) como a los Transformers puros (sin sensibilidad local adecuada).

### 4.4.3 Robustez y Escalabilidad

- **Robustez a desbalance de clases:** MarsSeg (Focal-Dice loss) > DeepLabV3+ (CE + GAN augmentation) > DepthFormer (WCE) > SegFormer (CE estándar). La clase Big Rock (~2%) y los Craters (<1%) son las más sensibles a la estrategia de pérdida; IC-TransUNet se prueba en este proyecto con CE + Dice + Focal para contrastar esta limitacion.
- **Escalabilidad computacional:** TerSeg (22M, 26G FLOPs) > SegFormer-B2 (25M, 62G FLOPs) > MarsSeg (N/R) > DeepLabV3+-R50 (~25M, ~177G FLOPs) > DeepLabV3+-R101 (~45M) > DepthFormer (~28M, resolución 2048×2048 → alto coste). IC-TransUNet añade una referencia dual-branch cuyo costo real se mide en el benchmark del proyecto.
- **Viabilidad en hardware embarcado de rovers:** TerSeg es el más viable (26G FLOPs, 22M params); DepthFormer requiere cómputo estéreo previo para generar el canal de profundidad, limitando su aplicabilidad en tiempo real.

### 4.4.4 Gaps Identificados en la Literatura

1. **Falta de benchmark unificado:** Cada paper reporta resultados en datasets distintos o subsets distintos de AI4MARS (M1/M2/M3 con diferentes criterios de aceptación), dificultando la comparación objetiva. Este proyecto contribuye precisamente con un benchmark unificado sobre el gold set M3.
2. **Clase Big Rock y arena como cuello de botella persistente:** Ningún modelo resuelve completamente la segmentación de clases raras (<5%) de manera robusta. TerSeg y MarsSeg avanzan en esta dirección pero en distintos datasets, sin resultados comparables.
3. **Ausencia de métricas de inferencia en tiempo real:** Solo TerSeg reporta FLOPs explícitos; ningún paper reporta latencia real en hardware de rover simulado o embarcado.
4. **Dominio de imágenes NavCam:** La mayoría de modelos se entrenan en imágenes NavCam (MSL). Las imágenes HazCam (mayor distorsión radial, gran angular) están subrepresentadas, siendo precisamente las usadas en la navegación de proximidad.
5. **Generalización entre misiones:** Ningún trabajo evalúa explícitamente la transferencia de MSL (Curiosity) a M2020 (Perseverance), a pesar de que las condiciones geológicas de Jezero Crater difieren notablemente de Gale Crater.
6. **Integración de modalidades adicionales:** DepthFormer es el único que incorpora profundidad. Otras modalidades como temperatura de superficie, datos espectrales o mapas de elevación orbital no han sido exploradas en este contexto.

---

*Selección final: SegFormer-B2 se adopta como propuesta original del proyecto por su balance Transformer jerarquico, estabilidad y eficiencia. Como benchmarks se seleccionan DeepLabV3+ (Mohammad et al., 2024), MarsSeg (Li et al., 2024), TerSeg (Fan et al., 2025), DepthFormer (Ma et al., 2025) e IC-TransUNet (Zhu et al., 2025/adaptacion propia), cubriendo el espectro CNN–Transformer–Híbrido relevante para la segmentación de terreno marciano sobre AI4MARS.*
