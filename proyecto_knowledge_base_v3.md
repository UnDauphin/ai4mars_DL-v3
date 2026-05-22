# Knowledge Base del Proyecto: Segmentación Semántica de Terreno Marciano

> **Entregable 2 — Deep Learning**
> Estado del Arte, EDA Mejorado e Implementación de Modelos Benchmark
>
> **Versión**: 3 — Refleja todas las decisiones tomadas en sesión de planificación extendida.
> Sustituye completamente la v2. Todo lo marcado como ~~tachado~~ o `[DEPRECADO]` en este documento ya no aplica.

---

## 1. Definición del Problema

### Objetivo
Desarrollar y comparar modelos de **segmentación semántica** (clasificación a nivel de píxel) capaces de identificar automáticamente el tipo de terreno en imágenes capturadas por rovers de la NASA en la superficie de Marte.

### Clases
| ID | Clase | Nota |
|----|-------|------|
| 0 | Soil (Suelo) | Clase mayoritaria |
| 1 | Bedrock (Roca base) | Clase mayoritaria |
| 2 | Sand (Arena) | Clase media |
| 3 | Big Rock (Rocas sueltas) | Clase minoritaria ~2% de píxeles |
| 255 | Ignore | Píxeles sin etiquetar — excluidos de la loss y métricas |

### Motivación crítica
Un error en la clasificación —por ejemplo, confundir arena profunda con suelo firme— puede dejar atrapado permanentemente al rover. La segmentación precisa es la base de la **navegación autónoma** en Marte.

---

## 2. Dataset — AI4MARS

| Campo | Valor |
|-------|-------|
| Fuente | Jet Propulsion Laboratory (JPL) / NASA |
| Disponibilidad | Zenodo ID: 15995036 |
| Total original | ~23.928 pares imagen-máscara |
| Tipo de imágenes | Escala de grises y RGB; resolución variable |
| Tipo de máscaras | PNG un canal, valores {0, 1, 2, 3, 255} |
| Etiquetado | Crowdsourcing (>326.000 etiquetas) + subconjunto "gold" validado por científicos JPL |

### Estructura real del dataset en disco

```
ai4mars_DL-v3/                              ← ROOT del proyecto
│
├── msl/                                    ← datos originales (no modificar)
│   └── ncam/
│       ├── images/
│       │   ├── edr/                        ← imágenes MSL train (JPEG)  ✅ USADO
│       │   ├── mxy/                        ← no usado
│       │   └── rng-30m/                    ← no usado
│       └── labels/
│           ├── train/                      ← máscaras MSL train (PNG)   ✅ USADO
│           └── test/
│               ├── masked-gold-min1-100agree/
│               ├── masked-gold-min2-100agree/
│               └── masked-gold-min3-100agree/  ← 322 máscaras  ✅ USADO (test fijo)
│   └── mcam/                               ← no usado
│
├── mer/                                    ← datos originales (no modificar)
│   ├── images/
│   │   ├── eff/                            ← imágenes MER (solo EDA exploratorio)
│   │   └── test/                           ← no usado
│   └── labels/
│       └── train/merged-unmasked/          ← máscaras MER (solo EDA exploratorio)
│
├── m2020/                                  ← datos originales (no modificar)
│   ├── images/ncam/                        ← imágenes M2020 (solo EDA exploratorio)
│   └── labels/NAV/                         ← máscaras M2020 (solo EDA exploratorio)
│
├── data/                                   ← generado por 02_preprocessing.ipynb ✅
│   ├── images_256/                         ← ~6.322 archivos .jpg (6k subset + 322 gold)
│   └── masks_256/                          ← ~6.322 archivos .png
│
├── processed/                              ← generado por notebooks 01 y 02 ✅
│   ├── manifest_clean.csv
│   ├── manifest_msl_train.csv
│   ├── manifest_msl_gold_test.csv
│   ├── manifest_msl_dominant.csv
│   ├── split_indices_msl6k.pkl             ← ⚠️ SUBIR AL REPO — mismo split para todos
│   ├── normalization_stats.json
│   └── class_weights.json
│
├── notebooks/                              ← TODOS los notebooks aquí
│   ├── 01_eda_exploratorio.ipynb           ✅ ejecutado
│   ├── 02_preprocessing.ipynb              ✅ ejecutado
│   ├── 03_eda_msl.ipynb                    ✅ ejecutado
│   ├── 04_marco_teorico.md                 ⚠️ pendiente completar
│   ├── 05a_model_deeplabv3plus.ipynb       ❌ pendiente
│   ├── 05b_model_segformer.ipynb           ❌ pendiente
│   ├── 05c_model_marsseg.ipynb             ❌ pendiente
│   ├── 05d_model_terseg.ipynb              ❌ pendiente
│   ├── 05e_model_depthformer.ipynb         ❌ pendiente
│   ├── 06_benchmark_estadistico.ipynb      ❌ pendiente
│   └── mars_utils.py                       ⚠️ pendiente reescritura v3
│
├── checkpoints/                            ← en .gitignore (se llena con los modelos)
│   └── {model}_seed{N}_best.pth
│
├── results/                                ← en .gitignore (se llena tras entrenamiento)
│   └── benchmark_results.csv
│
├── requirements.txt                        ✅
├── .gitignore                              ✅
├── changelog.md
├── info.md
├── label_keys.json
└── README.md
```

> ⚠️ **Cambio respecto a la estructura original del KB**: la carpeta `processed/` se movió de `notebooks/processed/` a la raíz del proyecto (`ROOT/processed/`). Todos los notebooks usan `PROCESSED_DIR = ROOT / "processed"`. No volver a colocarla dentro de `notebooks/`.

### Misiones incluidas en el dataset
| Misión | Rover | Cámara usada | Rol en el proyecto |
|--------|-------|--------------|-------------------|
| MSL | Curiosity | NavCam (edr) | **Entrenamiento, validación y test** |
| MER | Spirit / Opportunity | EFF | Solo EDA exploratorio (justificación de exclusión) |
| M2020 | Perseverance | NavCam | Solo EDA exploratorio (justificación de exclusión) |

### Gold set de test (MSL min3)
- **322 imágenes** con máscaras validadas por científicos del JPL
- Nivel de acuerdo: mínimo 3 etiquetadores al 100% (`masked-gold-min3-100agree`)
- **Es el test set fijo para todos los modelos** — nunca entra al entrenamiento ni validación
- Permite comparabilidad directa con Li et al. (2024) y Mohammad et al. (2024)
- Verificación de no-overlap con train pool: `assert len(set(train_ids) & set(gold_ids)) == 0`

---

## 3. Decisiones de Dataset y Preprocesamiento (v3)

### Decisión crítica: Solo subconjunto MSL — justificación empírica

**Se entrena exclusivamente con imágenes de la misión MSL (Curiosity NavCam).**

Esta decisión se justifica con evidencia empírica del `01_eda_exploratorio.ipynb`:

| Criterio | MSL | MER | M2020 | Decisión |
|----------|-----|-----|-------|----------|
| Tamaño | ~15.500 imgs (>70%) | ~6.500 imgs | ~471 imgs | MSL es el más grande |
| Domain shift (JS vs MSL) | 0.000 | >0.15 | >0.15 | JS alto → mezcla inapropiada sin domain adaptation |
| Consistencia de sensor | NavCam homogéneo | EFF (distinto) | NavCam (pocos datos) | MSL más consistente |
| Gold set estándar | 322 imgs JPL (min3) | Sin gold set comparable | Sin gold set | Solo MSL es comparable con literatura |
| Precedente | Li et al. (2024), Mohammad et al. (2024) | — | — | Maximiza comparabilidad |

**Argumento para el informe:**
> *"Siguiendo la metodología de Li et al. (2024), este trabajo restringe el entrenamiento a imágenes del rover Curiosity (MSL) para garantizar consistencia de dominio. El análisis cuantitativo de divergencia Jensen-Shannon entre misiones confirma domain shift significativo (JS > 0.15), lo que hace inapropiado mezclar misiones sin técnicas de domain adaptation explícitas. El conjunto de evaluación gold estándar (322 imágenes, min3-100agree) corresponde enteramente a MSL, maximizando la comparabilidad con la literatura existente."*

### Tamaño del subset de entrenamiento

**Decisión: 6.000 imágenes MSL seleccionadas aleatoriamente de forma estratificada**

| Opción | Pros | Contras | Decisión |
|--------|------|---------|----------|
| MSL completo (~15.500) | Máxima representatividad | ~45h/modelo × 3 seeds → inviable | ❌ |
| 8.000 imágenes | Buena cobertura | ~28h/modelo, muy ajustado | ⚠️ |
| **6.000 imágenes** | Balance tiempo/calidad | ~20h/modelo con 3 seeds | ✅ |
| 2.100 (versiones anteriores) | Muy rápido | Subset demasiado pequeño | ❌ `[DEPRECADO]` |

### Split (v3)

**Estructura del split:**
- Pool: 6.000 imágenes MSL del conjunto `train/` (estratificadas)
- **70% train (~4.200 imgs) / 30% val (~1.800 imgs)**
- Test: gold set fijo (322 imgs) — **no forma parte del pool de 6.000**

> ⚠️ **Cambio respecto a v2**: el split ya NO es 70/15/15 sobre el pool. El test set es el gold set fijo (min3). El pool de 6.000 se divide solo en train/val (70/30).

- **Persistencia**: `processed/split_indices_msl6k.pkl`
- **CRÍTICO**: Este archivo se genera una sola vez y se sube al repositorio. Todos los modelos deben usar exactamente el mismo split. No regenerar con otra semilla.

### Pipeline de preprocesamiento (v3)

El preprocesamiento está separado en su propio notebook (`02_preprocessing.ipynb`), que consume los manifests generados por `01_eda_exploratorio.ipynb`.

**Pasos:**
1. Leer `processed/manifest_msl_train.csv` (generado en notebook 01)
2. Seleccionar subset estratificado de 6.000 imágenes
3. Generar split train/val y guardar en `processed/split_indices_msl6k.pkl`
4. Resize imágenes y máscaras a **256×256 px**:
   - Imágenes: interpolación `BILINEAR` → guardadas como JPEG quality=95
   - Máscaras: interpolación `NEAREST` → guardadas como PNG (preserva valores exactos)
   - Output: `data/images_256/` y `data/masks_256/` (en ROOT)
5. Calcular stats de normalización **solo sobre el train set**:
   - Guardadas en `processed/normalization_stats.json`
6. Calcular pesos de clase **solo sobre el train set**:
   - Guardados en `processed/class_weights.json`

### Augmentaciones (solo train)
- Flip horizontal / vertical (p=0.5 cada uno)
- Rotación aleatoria ±15° — máscara rellena con `IGNORE_INDEX=255`, nunca con una clase
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
- Normalización: stats calculadas sobre train set (guardadas en `normalization_stats.json`)

> ⚠️ **Deprecado**: La normalización con stats de ImageNet `mean=[0.485, 0.456, 0.406]` se reemplaza por stats calculadas sobre el train set propio. Se documenta en `02_preprocessing.ipynb`.

---

## 4. Entorno de Desarrollo

### Configuración del entorno (desde cero)

**Prerrequisito**: NVIDIA Driver ≥ 596.36, CUDA máxima soportada: 13.2 (RTX 4050 laptop).

```powershell
# 1. Crear entorno conda
conda create -n mars_dl python=3.11 -y
conda activate mars_dl

# 2. Instalar PyTorch con CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Instalar dependencias del proyecto
pip install -r requirements.txt

# 4. Registrar como kernel de Jupyter
python -m ipykernel install --user --name mars_dl --display-name "Mars DL"
```

**Por qué Python 3.11**: Python 3.12 tiene incompatibilidades con `scikit-posthocs` y versiones de `timm` usadas en el proyecto. Python 3.11 es el punto dulce: maduro y compatible con todo el stack.

**Por qué CUDA 12.4**: El driver 596.36 soporta hasta CUDA 13.2, pero PyTorch 2.5.1 con cu124 es la combinación más estable disponible para RTX 4050. PyTorch embebe su propia versión de CUDA — no necesita match exacto con el driver.

### Verificación del entorno

```python
import torch
print(torch.__version__)              # debe ser 2.5.1+cu124
print(torch.cuda.is_available())      # debe ser True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4050 Laptop GPU

import transformers, timm, numpy, pandas, sklearn, PIL, seaborn
print("Todas las dependencias OK")
```

**Resultado verificado en sesión de instalación**:
```
2.5.1+cu124
True
NVIDIA GeForce RTX 4050 Laptop GPU
Todas las dependencias OK
```

### Hardware del proyecto

| Recurso | GPU | VRAM | TDP | Rol |
|---------|-----|------|-----|-----|
| Laptop principal | RTX 4050 Laptop | 6GB | 35W | SegFormer-B2, MarsSeg |
| Laptop compañero 1 | RTX 4050 Laptop | 6GB | 35W | TerSeg |
| Laptop compañero 2 | RTX 4050 Laptop | 6GB | 35W | DepthFormer-RGB |
| Google Colab Pro | T4 / A100 | 15–40GB | — | DeepLabV3+ |

> ⚠️ **Corregido respecto a v2**: El TDP es 35W (no 50W). Confirmado por `nvidia-smi` (`Pwr:Usage/Cap: 1W / 35W`).

---

## 5. Requirements (`requirements.txt`)

```
# ── Deep Learning ────────────────────────────────────────────────────────────
# Instalar PyTorch por separado con:
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
transformers>=4.40.0
timm>=1.0.26

# ── Ciencia de datos ─────────────────────────────────────────────────────────
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.13.0
scikit-learn>=1.5.0
scikit-posthocs>=0.9.0

# ── Visión e imágenes ────────────────────────────────────────────────────────
Pillow>=10.3.0
matplotlib>=3.9.0
seaborn>=0.13.0

# ── Utilidades ───────────────────────────────────────────────────────────────
tqdm>=4.66.0

# ── Notebooks y Jupyter Book ─────────────────────────────────────────────────
jupyter>=1.0.0
jupyterlab>=4.0.0
jupyter-book>=1.0.0
ipywidgets>=8.1.0
ipykernel>=6.29.0
```

> **Nota para compañeros**: PyTorch NO está en el requirements.txt porque requiere la URL especial del índice de CUDA. Instálalo primero con el comando de la sección 4.

---

## 6. `.gitignore`

```gitignore
# ── Datos originales (demasiado grandes para git) ────────────────────────────
msl/
mer/
m2020/
data/

# ── Checkpoints (archivos de varios GB) ──────────────────────────────────────
checkpoints/

# ── Resultados intermedios ────────────────────────────────────────────────────
results/

# ── Processed: ignorar todo EXCEPTO el split fijo ────────────────────────────
processed/*
!processed/split_indices_msl6k.pkl

# ── Entornos Python ───────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv
env/

# ── Jupyter ───────────────────────────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb_checkpoints

# ── Archivos del sistema ──────────────────────────────────────────────────────
.DS_Store
Thumbs.db
desktop.ini

# ── IDEs ──────────────────────────────────────────────────────────────────────
.vscode/
.idea/
```

> **Qué SÍ se sube al repo**: `processed/split_indices_msl6k.pkl`, `requirements.txt`, todos los notebooks `.ipynb`, `mars_utils.py`, `04_marco_teorico.md`, `.gitignore`, `README.md`.

---

## 7. Flujo de Trabajo entre Notebooks

```
01_eda_exploratorio.ipynb
  └── produce: processed/manifest_clean.csv
               processed/manifest_msl_train.csv      ─┐
               processed/manifest_msl_gold_test.csv   │
                                                      ▼
              02_preprocessing.ipynb
                └── produce: data/images_256/
                             data/masks_256/
                             processed/split_indices_msl6k.pkl   ─┐
                             processed/normalization_stats.json   │
                             processed/class_weights.json         │
                             processed/manifest_msl_dominant.csv  │
                                                                  ▼
                            03_eda_msl.ipynb
                              └── (análisis sobre datos procesados)
                                                                  │
                                                                  ▼
                            05a–05e notebooks de modelos
                              └── consumen: data/images_256/, data/masks_256/
                                           processed/split_indices_msl6k.pkl
                                           processed/normalization_stats.json
                                           processed/class_weights.json
                                  producen: checkpoints/
                                            results/benchmark_results.csv
                                                                  │
                                                                  ▼
                            06_benchmark_estadistico.ipynb
                              └── consume: results/benchmark_results.csv
```

---

## 8. Infraestructura Común (`mars_utils.py`) — pendiente reescritura v3

> **Estado**: pendiente. El archivo `mars_utils.py` necesita actualizarse para reflejar la nueva estructura de directorios (`processed/` en ROOT) y el nuevo split.

### Constantes clave (v3)
```python
NUM_CLASSES  = 4
IGNORE_INDEX = 255
IMG_SIZE     = 256
BATCH_SIZE   = 16
SEEDS        = [42, 123, 7]
N_SUBSET     = 6000   # imágenes MSL en el pool train+val

# Rutas — todas relativas a ROOT
ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed"
DATA_DIR      = ROOT / "data"
IMAGES_256    = DATA_DIR / "images_256"
MASKS_256     = DATA_DIR / "masks_256"
```

### Componentes requeridos
- `MarsTerrainDataset`: Dataset PyTorch — lee desde `data/images_256/` y `data/masks_256/`
- `MetricsAccumulator`: TP/FP/FN para mIoU y pixel accuracy por clase
- `train_one_epoch` / `evaluate`: loops con AMP, retornan loss Y mIoU
- `train_model`: loop completo con early stopping (patience=7), checkpoint, historial train+val
- `plot_training_curves`: curvas loss y mIoU — solo `plt.show()`, sin `savefig`
- `run_multi_seed`: 3 seeds, reporta media ± std e IC95%
- `append_benchmark_results`: guarda en `results/benchmark_results.csv`
- `visualize_predictions`: predicciones vs ground truth

### Losses disponibles
| Clase | Descripción |
|-------|-------------|
| `nn.CrossEntropyLoss` | CE estándar con `ignore_index=255` |
| `FocalLoss` | α=0.25, γ=2.0 |
| `DiceLoss` | Para clases raras |
| `FocalDiceLoss` | Combinación Focal + Dice |

---

## 9. Modelos Benchmark (v3)

### Configuración experimental común
- **Seeds**: [42, 123, 7] — obligatorio para análisis estadístico
- **Max epochs**: 80
- **Early stopping**: patience=7 sobre val mIoU
- **Batch size**: 16
- **Input**: `[B, 3, 256, 256]` — canal grayscale replicado ×3
- **Métrica de selección de checkpoint**: val mIoU
- **Test**: evaluación sobre gold set fijo (manifest_msl_gold_test.csv)

### Distribución entre máquinas

| Máquina | Modelo | Tiempo estimado |
|---------|--------|-----------------|
| Laptop principal | SegFormer-B2 | ~11h |
| Laptop principal (2ª ronda) | MarsSeg | ~15h |
| Compañero 1 | TerSeg | ~20h |
| Compañero 2 | DepthFormer-RGB | ~12h |
| Colab Pro | DeepLabV3+ | ~4h en A100 |

### 9.1 DeepLabV3+ (ResNet-50) — `05a_model_deeplabv3plus.ipynb`

**Paper**: Mohammad et al., iSpaRo 2024. DOI: 10.1109/iSpaRo60631.2024.10687827

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | ResNet-50 preentrenado ImageNet |
| Parámetros | 42.00M |
| Loss | CrossEntropyLoss (ignore_index=255) |
| Optimizer | SGD (lr=0.001, momentum=0.9, wd=1e-4) |
| Scheduler | PolynomialLR (power=0.9, 80 iters) |
| Aux weight | 0.4 |
| Batch size | 16 |
| Entrenamiento | Colab Pro (A100/T4) |

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.6806 ± 0.0107   seed: 0.6655 | 0.6893 | 0.6869
```

### 9.2 SegFormer-B2 — `05b_model_segformer.ipynb`

**Paper**: Xie et al., NeurIPS 2021. arXiv:2105.15203

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | MiT-B2 (HuggingFace: nvidia/mit-b2) |
| Parámetros | 27.35M |
| Loss | CrossEntropyLoss (ignore_index=255) |
| Optimizer | AdamW (lr=6e-5, wd=0.01, betas=(0.9, 0.999)) |
| Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| Salida | H/4×W/4 → upsample bilinear a 256×256 |
| Entrenamiento | Laptop principal |

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.7592 ± 0.0078   seed: 0.7547 | 0.7701 | 0.7526
```

### 9.3 MarsSeg — `05c_model_marsseg.ipynb`

**Paper**: Li et al., arXiv:2404.04155, Beihang University 2024.

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | ResNet-50 + MiniASPP + PSA + SPPM |
| Parámetros | 38.61M |
| Loss | FocalDiceLoss (α=0.25, γ=2.0) |
| Optimizer | SGD (lr=0.001, momentum=0.9, wd=1e-4) |
| Scheduler | PolynomialLR (power=0.9, 30 iters) |
| Dilation | layer4 con dilation=2 |
| Entrenamiento | Laptop principal (2ª ronda) |

**Advertencia**: Alta varianza en big_rock (±0.054) — el más inestable. Requiere supervisión.

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.6601 ± 0.0122   seed: 0.6651 | 0.6719 | 0.6434
```

### 9.4 TerSeg (Dual-Branch CNN + Swin) — `05d_model_terseg.ipynb`

**Paper**: Fan et al., Expert Systems with Applications 270 (2025) 126397. DOI: 10.1016/j.eswa.2025.126397

| Hiperparámetro | Valor |
|----------------|-------|
| Branch CNN | ResNet-34 (pretrained) |
| Branch Transformer | Swin-Tiny (timm, pretrained) |
| Parámetros | 49.19M |
| Loss | FocalLoss (α=0.25, γ=2.0) |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | ReduceLROnPlateau (mode='max', patience=5, factor=0.5) |
| Entrenamiento | Compañero 1 |

**Nota**: TerSeg interpola a 224×224 antes de Swin (limitación del modelo pretrained).

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.7498 ± 0.0089   seed: 0.7476 | 0.7617 | 0.7401
```

### 9.5 DepthFormer-RGB (Swin-Tiny + PPM) — `05e_model_depthformer.ipynb`

**Paper**: Ma et al., Earth and Space Science 12, e2024EA003812 (2025). DOI: 10.1029/2024EA003812

> **Variante implementada**: DepthFormer-RGB — sin canal de profundidad (AI4MARS no tiene mapas de profundidad).

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | Swin-Tiny (timm, pretrained, img_size=256) |
| Parámetros | 31.58M |
| Loss | CrossEntropyLoss con class weights [1.0, 0.8, 2.0, 8.0] |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | PolynomialLR (power=0.9, 30 iters) |
| Entrenamiento | Compañero 2 |

**Nota**: Swin-Tiny configurado con img_size=256 para evitar interpolación posicional.

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.7609 ± 0.0123   seed: 0.7777 | 0.7564 | 0.7485
```

---

## 10. Otros Modelos Identificados en la Literatura (No Implementados)

| Modelo | Paper | mIoU AI4MARS | Por qué no se implementó |
|--------|-------|-------------|--------------------------|
| CPNet | Yu et al., CVPR 2020 | 80.01% | Tiempo limitado |
| PSPNet | Zhao et al., CVPR 2017 | 78.44% | Descartado por mantener DeepLabV3+ |
| U-Net | Ronneberger et al., 2015 | ~67% | Inferior a SegFormer en este contexto |
| SegMarsViT | Dai et al., 2022 | 68.4% | Inferior a modelos seleccionados |
| LMFFNet | Shi et al., 2023 | 69.47% | Orientado a deployment, no al benchmark |
| Mask2Former | Cheng et al., 2022 | 75.84%* | No específico de Marte |
| SSL Teacher-Student | Zhang et al., 2023 | 75% | Paradigma semi-supervisado distinto |

*En DepthMars dataset, no AI4MARS directamente.

---

## 11. Estado del Arte — Síntesis

### Papers del benchmark (5 seleccionados)

| Paper | Año | Tipo | Dataset principal | mIoU reportado |
|-------|-----|------|-------------------|----------------|
| Mohammad et al. (DeepLabV3+) | 2024 | CNN | AI4MARS M3 | 88% (con GAN aug) |
| Li et al. (MarsSeg) | 2024 | CNN avanzado | Mars-Seg MSL | 65.69% |
| Fan et al. (TerSeg) | 2025 | CNN+Transformer | S⁵mars | 71.96% |
| Ma et al. (DepthFormer) | 2025 | Transformer | DepthMars | 75.99% |
| Xie et al. (SegFormer-B2) | 2021 | Transformer | AI4MARS | 83.55% |

### Tendencias clave
- CNNs clásicos (DeepLabV3+) dominan en AI4MARS estándar con GAN augmentation
- Transformers puros (SegFormer) son eficientes pero degradan en datasets multi-clase complejos
- Híbridos CNN+Transformer (TerSeg, DepthFormer) representan el estado del arte 2024-2025
- La clase `big_rock` (~2% de píxeles) es el cuello de botella universal en todos los modelos

### Gaps en la literatura
1. No existe benchmark unificado (cada paper usa datasets distintos)
2. `big_rock` y arena siguen sin resolverse completamente
3. Ningún paper reporta latencia real en hardware de rover
4. No hay evaluación de generalización entre misiones (MSL → Perseverance)
5. Solo DepthFormer explora modalidades adicionales (profundidad)

---

## 12. Requisitos del Entregable y Estado Actual

| Criterio | Peso | Estado |
|----------|------|--------|
| EDA Mejorado | 25% | ✅ Completo — notebooks 01, 02 y 03 ejecutados y verificados |
| Revisión de literatura | 25% | ⚠️ `04_marco_teorico.md` — existe, revisar y completar |
| Implementación modelos | 30% | ❌ Pendiente — notebooks 05a a 05e |
| Benchmark y análisis estadístico | 20% | ❌ Pendiente — requiere resultados de los 5 modelos |

### Benchmark estadístico — requisitos no negociables

- Pruebas pareadas: t-test pareado (normalidad) o Wilcoxon signed-rank (no paramétrico)
- Múltiples modelos: Friedman test + post-hoc Nemenyi o Bonferroni/Holm
- Nivel de significancia: α = 0.05
- P-valores exactos (no solo "< 0.05")
- Verificación de supuestos: Shapiro-Wilk para normalidad
- **Mínimo 3 runs por modelo** — no negociable

Estructura de datos requerida tras entrenamiento:
```python
resultados = {
    'DeepLabV3+':      [seed42, seed123, seed7],   # mIoU sobre gold set
    'SegFormer-B2':    [seed42, seed123, seed7],
    'MarsSeg':         [seed42, seed123, seed7],
    'TerSeg':          [seed42, seed123, seed7],
    'DepthFormer-RGB': [seed42, seed123, seed7],
}
```

---

## 13. Notas de Implementación

- **AMP siempre activo**: `GradScaler` creado una sola vez fuera del loop de epochs
- **Monitoreo de overfitting**: registrar train_loss Y val_loss por epoch
- **Data leakage verificado**: `assert len(set(train_ids) & set(gold_ids)) == 0` en notebook 01
- **Masks fill en rotación**: siempre `IGNORE_INDEX=255`, nunca con una clase válida
- **Split compartido**: `processed/split_indices_msl6k.pkl` se sube al repo — no regenerar
- **Checkpoints en Drive**: crítico en Colab Pro para no perder progreso
- **Figuras**: solo `plt.show()` en todos los notebooks — sin `savefig` local
- **Rutas**: siempre relativas a ROOT (`NOTEBOOK_DIR.parent`) — sin rutas absolutas
- **Normalización**: stats calculadas solo sobre train set, guardadas en JSON, aplicadas a val y gold test
- **PROCESSED_DIR**: siempre `ROOT / "processed"` — nunca `NOTEBOOK_DIR / "processed"`
- **Bug conocido en 03_eda_msl.ipynb**: la línea `rec["big_rock_border_px"] = int((np.abs(br_neighbor_y) + np.abs(br_neighbor_x)).sum())` lanza `ValueError` por shapes incompatibles `(255,256)` y `(256,255)`. Corrección: `int(np.abs(br_neighbor_y).sum() + np.abs(br_neighbor_x).sum())`

---

## 14. Instrucciones para Compañeros (onboarding)

1. **Clonar el repositorio** desde el link que se compartirá por el grupo
2. **Descargar los datos** desde Zenodo ID: 15995036 y colocarlos en la raíz del proyecto respetando la estructura de carpetas
3. **Crear el entorno** siguiendo la sección 4 de este documento
4. **Ejecutar en orden**:
   - `01_eda_exploratorio.ipynb` — genera los manifests en `processed/`
   - `02_preprocessing.ipynb` — genera imágenes resizadas en `data/` y artefactos en `processed/`
   - El notebook del modelo asignado (`05a` a `05e`)
5. **NO regenerar** `processed/split_indices_msl6k.pkl` — usar el que viene en el repo
6. **Guardar checkpoints en Google Drive** si se usa Colab Pro
7. **Verificar que `processed/` está en la raíz del proyecto** (ROOT), no dentro de `notebooks/`

---

*Documento actualizado — v3. Reemplaza completamente la v2.*
*Últimos cambios: instalación y verificación del entorno completadas; notebooks 02 y 03 ejecutados; `processed/` movida a ROOT; `.gitignore` actualizado; bug de shapes en notebook 03 documentado y corregido.*
*Próximos pasos: (1) `04_marco_teorico.md`, (2) notebooks de modelos 05a–05e en paralelo entre compañeros.*
