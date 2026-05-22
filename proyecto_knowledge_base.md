# Knowledge Base del Proyecto: Segmentación Semántica de Terreno Marciano

> **Entregable 2 — Deep Learning**
> Estado del Arte, EDA Mejorado e Implementación de Modelos Benchmark
>
> **Versión**: 8 — Fix del sistema de checkpoints: history ahora se persiste en `_periodic.pth` y `_progress.json`; intervalo de checkpoint cambiado de 5 a 3 epochs; instrucciones para compañeros actualizadas; Bug 10 documentado. Sustituye completamente la v7.

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
├── data/                                   ← generado por 02_preprocessing.ipynb ✅ EN EL REPO
│   ├── images_256/                         ← 6.322 archivos .jpg (6k subset + 322 gold) ~137MB
│   └── masks_256/                          ← 6.322 archivos .png
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
│   ├── 04_marco_teorico.md                 ✅ completado (v3 corregida)
│   ├── 05a_model_deeplabv3plus.ipynb       ⚠️ generado y debugueado — pendiente entrenamiento completo (Colab Pro)
│   ├── 05b_model_segformer.ipynb           ✅ generado — pendiente entrenamiento completo
│   ├── 05c_model_marsseg.ipynb             ⚠️ debugueado manualmente — pendiente entrenamiento completo
│   ├── 05d_model_terseg.ipynb              ⚠️ debugueado y validado fast_subset=True — pendiente entrenamiento completo
│   ├── 05e_model_depthformer.ipynb         ⚠️ debugueado y validado fast_subset=True — pendiente entrenamiento completo
│   ├── 06_benchmark_estadistico.ipynb      ❌ pendiente
│   └── mars_utils.py                       ✅ reescrito v4
│
├── checkpoints/                            ← en .gitignore (se llena con los modelos)
│   └── {model}_seed{N}_best.pth
│
├── results/                                ← EN EL REPO (se sube por cada companero via PR)
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

> ⚠️ **Deprecado**: La normalización con stats de ImageNet `mean=[0.485, 0.456, 0.406]` se reemplaza por stats calculadas sobre el train set propio.

---

## 4. Entorno de Desarrollo

### Configuración del entorno (desde cero)

**Prerrequisito**: NVIDIA Driver ≥ 596.36, CUDA máxima soportada: 13.2 (RTX 4050 laptop).

```powershell
# 1. Crear entorno conda
conda create -n mars_dl python=3.11 -y
conda activate mars_dl

# 2. Instalar PyTorch con CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Instalar dependencias del proyecto
pip install -r requirements.txt

# 4. Registrar como kernel de Jupyter
python -m ipykernel install --user --name mars_dl --display-name "Mars DL"
```

**Por qué Python 3.11**: Python 3.12 tiene incompatibilidades con `scikit-posthocs` y versiones de `timm` usadas en el proyecto.

**Por qué CUDA 12.4**: El driver 596.36 soporta hasta CUDA 13.2, pero PyTorch 2.6.0 con cu124 es la combinación más estable disponible para RTX 4050. PyTorch < 2.6 es incompatible con versiones actuales de `transformers` (CVE-2025-32434).

### Verificación del entorno

```python
import torch
print(torch.__version__)              # debe ser 2.6.0+cu124
print(torch.cuda.is_available())      # debe ser True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4050 Laptop GPU

import transformers, timm, numpy, pandas, sklearn, PIL, seaborn
print("Todas las dependencias OK")
```

### Hardware del proyecto

| Recurso | GPU | VRAM | TDP | Modelo asignado |
|---------|-----|------|-----|-----------------|
| Laptop principal | RTX 4050 Laptop | 6GB | 50W | SegFormer-B2 |
| Laptop compañero 1 | RTX 4050 Laptop | 6GB | 50W | TerSeg |
| Laptop compañero 2 | RTX 3050 Laptop | 6GB | 50W | DepthFormer-RGB, MarsSeg |
| Google Colab Pro | T4 / A100 | 15–40GB | — | DeepLabV3+ |

> ⚠️ **Nota**: El TDP real es 50W (máximo con cargador conectado). `nvidia-smi` puede reportar 35W si el equipo está desconectado de la corriente, ya que en ese modo el wattage se limita automáticamente. El compañero 2 tiene RTX 3050 en vez de 4050 — misma VRAM (6GB) pero menor rendimiento de cómputo; se le asignan DepthFormer (~31M params) y MarsSeg (~38M params) por ser más ligeros que TerSeg (~49M params).

---

## 5. Requirements (`requirements.txt`)

```
# ── Deep Learning ────────────────────────────────────────────────────────────
# Instalar PyTorch por separado con:
# pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
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

> **Nota para compañeros**: PyTorch NO está en el requirements.txt porque requiere la URL especial del índice de CUDA.

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

## 8. Infraestructura Común — `mars_utils.py` (v4)

> **Estado**: ✅ Reescrito y finalizado en v4. Ubicación: `notebooks/mars_utils.py`.

### Ubicación y uso

El archivo vive en `notebooks/mars_utils.py` — mismo nivel que los notebooks `05a` a `05e`.
Import desde cualquier notebook:

```python
from mars_utils import *
```

### Compatibilidad local / Google Colab

`ROOT` se detecta automáticamente en ambos entornos mediante `_find_root()`:

```python
def _find_root() -> Path:
    # En local: ROOT es parent.parent de este archivo
    local = Path(__file__).resolve().parent.parent
    if (local / "processed").exists():
        return local
    # En Colab: buscar desde el directorio de trabajo actual
    cwd = Path(os.getcwd())
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "processed").exists():
            return candidate
    raise RuntimeError("No se encontró ROOT del proyecto.")
```

En Colab, antes del import hay que montar Drive y añadir al path:

```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/ai4mars_DL-v3/notebooks')
from mars_utils import *
```

### Constantes clave

```python
NUM_CLASSES  = 4
IGNORE_INDEX = 255
IMG_SIZE     = 256
BATCH_SIZE   = 16
SEEDS        = [42, 123, 7]

CLASS_NAMES  = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
```

### Normalización

**No usa stats de ImageNet.** Carga desde `processed/normalization_stats.json` (calculadas sobre el train set):

```python
mean, std = load_norm_stats()
```

### Funciones disponibles

| Función | Descripción |
|---------|-------------|
| `set_seed(seed)` | Fija semilla en random, numpy, torch y CUDA |
| `load_norm_stats()` | Carga mean/std del train set desde JSON |
| `load_split()` | Carga df_train, df_val, df_gold — nunca regenera el split |
| `make_fast_subset(df_train, df_val, n_train, n_val)` | Subset estratificado pequeño para pruebas rápidas |
| `build_dataloaders(df_train, df_val, df_gold, mean, std)` | Construye los 3 DataLoaders |
| `JointTransformTrain(mean, std)` | Augmentaciones para train (flip, rotación, color jitter) |
| `JointTransformVal(mean, std)` | Solo ToTensor + Normalize (para val y gold) |
| `MarsTerrainDataset(df, transform)` | Dataset PyTorch — devuelve `(img, mask)` |
| `MetricsAccumulator` | Acumula TP/FP/FN, calcula mIoU y pixel accuracy |
| `DiceLoss` | Dice loss con ignore_index |
| `FocalLoss(alpha, gamma)` | Focal loss con ignore_index |
| `FocalDiceLoss(alpha, gamma)` | Combinación Focal + Dice |
| `train_one_epoch(...)` | Un epoch de entrenamiento con AMP |
| `evaluate(...)` | Evaluación sin gradientes con AMP |
| `train_model(...)` | Loop completo con early stopping y checkpointing |
| `run_multi_seed(...)` | 3 seeds, evalúa en gold set, devuelve summary |
| `append_benchmark_results(summary, params_M)` | Agrega fila al CSV de benchmark |
| `plot_training_curves(history, model_name)` | Curvas loss y mIoU — solo plt.show() |
| `plot_best_seed_curves(summary)` | Curvas del seed con mejor mIoU en gold test |
| `print_summary_table(summary)` | Tabla resumen agregada (media ± std, IC95%) |
| `visualize_predictions(model, df_gold, device, mean, std, n)` | n ejemplos imagen / GT / predicción |
| `count_parameters(model)` | Parámetros entrenables en millones |

### Pruebas rápidas vs producción

`run_multi_seed` tiene el parámetro `fast_subset`:

```python
# Prueba rápida en tu PC (~200 imgs train, ~50 val, ~2 min/seed)
summary = run_multi_seed(..., fast_subset=True, n_train_fast=200, n_val_fast=50)

# Producción (compañeros, entrenamiento completo)
summary = run_multi_seed(..., fast_subset=False)
```

### Resultados: CSV, no JSON

Los resultados del benchmark se guardan en `results/benchmark_results.csv`.
Los archivos JSON (`normalization_stats.json`, `class_weights.json`) son artefactos de configuración del preprocesamiento — no de resultados.

### Flujo típico en cada notebook de modelo

```python
from mars_utils import *

# 1. Cargar datos
df_train, df_val, df_gold = load_split()
mean, std = load_norm_stats()

# 2. Definir modelo, loss, optimizer, scheduler (específico de cada notebook)
def build_model(): ...
def criterion_fn(): ...
def optimizer_fn(params): ...
def scheduler_fn(optimizer): ...

# 3. Entrenar
summary = run_multi_seed(
    model_fn=build_model, df_train=df_train, df_val=df_val, df_gold=df_gold,
    criterion_fn=criterion_fn, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
    model_name="NombreModelo", device=DEVICE,
    fast_subset=False,   # True para pruebas, False para producción
)

# 4. Resultados
print_summary_table(summary)
plot_best_seed_curves(summary)
append_benchmark_results(summary, params_M=count_parameters(build_model()))

# 5. Visualización cualitativa
visualize_predictions(mejor_modelo, df_gold, DEVICE, mean=mean, std=std, n=5)
```

---

## 9. Modelos Benchmark (v4)

### Configuración experimental común
- **Seeds**: [42, 123, 7] — obligatorio para análisis estadístico
- **Max epochs**: 80
- **Early stopping**: patience=7 sobre val mIoU
- **Batch size**: 16
- **Input**: `[B, 3, 256, 256]` — canal grayscale replicado ×3
- **Métrica de selección de checkpoint**: val mIoU
- **Test**: evaluación sobre gold set fijo (manifest_msl_gold_test.csv)
- **Resultados**: `results/benchmark_results.csv` (una fila por modelo)
- **Visualización por notebook**: tabla resumen agregada + curva del mejor seed

### Orden de implementación y distribución

| Orden | Notebook | Modelo | Máquina | Tiempo estimado |
|-------|----------|--------|---------|-----------------|
| 1 | 05b | SegFormer-B2 | Laptop principal | ~11h |
| 2 | 05c | MarsSeg | Laptop principal (2ª ronda) | ~15h |
| 3 | 05d | TerSeg | Compañero 1 | ~20h |
| 4 | 05e | DepthFormer-RGB | Compañero 2 | ~12h |
| 5 | 05a | DeepLabV3+ | Colab Pro | ~4h en A100 |

> **Nota**: El orden de implementación empieza por SegFormer (no DeepLabV3+) porque es el modelo de la laptop principal y permite verificar el pipeline antes de distribuir a los compañeros. DeepLabV3+ va en Colab y puede correr en paralelo.

### Arquitecturas — decisiones clave por modelo

Todas las arquitecturas están definidas. No hay ambigüedad pendiente.

| Modelo | Backbone | Decoder | Decisión especial |
|--------|----------|---------|-------------------|
| DeepLabV3+ | ResNet-50 (ImageNet) | ASPP | Aux head con weight=0.4 |
| SegFormer-B2 | MiT-B2 (HuggingFace) | MLP ligero | Salida H/4×W/4 → upsample bilinear a 256×256 via wrapper |
| MarsSeg | ResNet-50 + MiniASPP + PSA + SPPM | Upsampling ×2 | layer4 con dilation=2 |
| TerSeg | ResNet-34 + Swin-Tiny (timm) | FLGA | Interpola a 224×224 antes de Swin (limitación pretrained) |
| DepthFormer-RGB | Swin-Tiny (timm, img_size=256) | UPerNet + PPM | Sin canal de profundidad — AI4MARS no tiene depth maps |

### 9.1 DeepLabV3+ (ResNet-50) — `05a_model_deeplabv3plus.ipynb`

**Paper**: Mohammad et al., iSpaRo 2024. DOI: 10.1109/iSpaRo60631.2024.10687827

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | ResNet-50 preentrenado ImageNet |
| Parámetros | ~42.00M |
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

**Estado**: ✅ Notebook generado.

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | MiT-B2 (HuggingFace: `nvidia/mit-b2`, pretrained) |
| Parámetros | ~27.35M |
| Loss | CrossEntropyLoss (ignore_index=255) |
| Optimizer | AdamW (lr=6e-5, wd=0.01, betas=(0.9, 0.999)) |
| Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| Salida | H/4×W/4 → upsample bilinear a 256×256 |
| Entrenamiento | Laptop principal |

**Implementación**: `SegFormerForSemanticSegmentation` de HuggingFace envuelto en `SegFormerWrapper(nn.Module)` que aplica `F.interpolate` a resolución completa. El wrapper vive en el notebook, no en `mars_utils.py` (cada modelo define su arquitectura localmente).

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.7592 ± 0.0078   seed: 0.7547 | 0.7701 | 0.7526
```

### 9.3 MarsSeg — `05c_model_marsseg.ipynb`

**Paper**: Li et al., arXiv:2404.04155, Beihang University 2024.

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | ResNet-50 + MiniASPP + PSA + SPPM |
| Parámetros | ~38.61M |
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
| Parámetros | ~49.19M |
| Loss | FocalLoss (α=0.25, γ=2.0) |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | ReduceLROnPlateau (mode='max', patience=5, factor=0.5) |
| Entrenamiento | Compañero 1 |

**Nota**: TerSeg interpola a 224×224 antes de Swin (limitación del modelo pretrained).

**Resultados de referencia** (subset 2.100 imgs — `[DEPRECADO]`, solo referencia histórica):
```
mIoU = 0.7498 ± 0.0089   seed: 0.7476 | 0.7617 | 0.7401
```

### 9.5 DepthFormer-RGB (Swin-Tiny + UPerNet + PPM) — `05e_model_depthformer.ipynb`

**Paper**: Ma et al., Earth and Space Science 12, e2024EA003812 (2025). DOI: 10.1029/2024EA003812

> **Variante implementada**: DepthFormer-RGB — sin canal de profundidad.
> **Justificación**: AI4MARS no tiene depth maps disponibles. Esta variante replica exactamente la rama de ablación que el propio paper reporta en su Tabla 3 (Swin Transformer base vs DepthFormer), lo que hace la comparación académicamente válida.
> **Descartado**: uso de MiDaS/Depth Anything v2 para estimar depth maps sintéticos — añade complejidad sin garantía de mejora y sale del scope del entregable.

| Hiperparámetro | Valor |
|----------------|-------|
| Backbone | Swin-Tiny (timm, pretrained, img_size=256) |
| Decoder | UPerNet + PPM |
| Parámetros | ~31.58M |
| Loss | CrossEntropyLoss con class weights [1.0, 0.8, 2.0, 8.0] |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | PolynomialLR (power=0.9, 30 iters) |
| Entrenamiento | Compañero 2 |

**Nota**: Swin-Tiny configurado con `img_size=256` para evitar interpolación posicional.

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
| Novel DNN (Zhang et al., 2018) | arXiv:1808.08395 | — | Modelo de navegación visual, no segmentación semántica |

*En DepthMars dataset, no AI4MARS directamente.

---

## 11. Marco Teórico — Estado

**Archivo**: `notebooks/04_marco_teorico.md`
**Estado**: ✅ Completado y corregido (v3 del documento).

### Correcciones aplicadas respecto a la versión anterior
1. Renumerado de `03_` a `04_` y secciones de 3.x a 4.x
2. Novel DNN (Zhang et al., 2018) eliminado del top de modelos de segmentación — es modelo de navegación, no segmentación
3. MarsNet eliminado — sin paper verificado
4. MarsSeg: aclaración de que mIoU 80.89 es IoU de Big Rock, no mIoU global
5. TerSeg: corregido de "10 categorías" a "9 categorías" (Sky, Ridge, Soil, Sand, Bedrock, Rock, Rover, Trace, Hole)
6. SegFormer: cita académica correcta (Xie et al., 2021) + mención del repositorio GitHub solo como evidencia de aplicabilidad al dominio marciano, no como fuente de métricas
7. Sección Paper Guía: pendiente — se añadirá cuando se defina el modelo original

---

## 12. Estado del Arte — Síntesis

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

## 13. Requisitos del Entregable y Estado Actual

| Criterio | Peso | Estado |
|----------|------|--------|
| EDA Mejorado | 25% | ✅ Completo — notebooks 01, 02 y 03 ejecutados y verificados |
| Revisión de literatura | 25% | ✅ `04_marco_teorico.md` completado y corregido |
| Implementación modelos | 30% | ⚠️ En progreso — `05b_model_segformer.ipynb` ejecutado y validado (fast_subset); 05a, 05c, 05d, 05e pendientes entrenamiento completo |
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

### Lo que muestra cada notebook de modelo
- **Tabla resumen agregada**: mIoU mean ± std, IC95%, IoU por clase (4 clases), pixel accuracy, tiempo medio, epoch mejor
- **Curva de entrenamiento**: loss y mIoU de train y val — solo del seed con mejor mIoU en gold test
- **Visualización cualitativa**: 5 ejemplos imagen / GT / predicción del gold set, usando el checkpoint del mejor seed
- **CSV**: una fila añadida a `results/benchmark_results.csv` con `append_benchmark_results()`

---

## 14. Notas de Implementación

- **AMP siempre activo**: `GradScaler` creado una sola vez fuera del loop de epochs
- **Monitoreo de overfitting**: registrar train_loss Y val_loss por epoch
- **Data leakage verificado**: `assert len(set(train_ids) & set(gold_ids)) == 0` en `load_split()`
- **Masks fill en rotación**: siempre `IGNORE_INDEX=255`, nunca con una clase válida
- **Split compartido**: `processed/split_indices_msl6k.pkl` se sube al repo — no regenerar
- **Checkpoints en Drive**: crítico en Colab Pro para no perder progreso
- **Figuras**: solo `plt.show()` en todos los notebooks — sin `savefig` local
- **Rutas**: ROOT detectado automáticamente por `_find_root()` en `mars_utils.py`
- **Normalización**: stats calculadas solo sobre train set, guardadas en JSON, aplicadas a val y gold test
- **PROCESSED_DIR**: siempre `ROOT / "processed"` — nunca `NOTEBOOK_DIR / "processed"`
- **Arquitecturas**: cada notebook define su propia clase de modelo localmente — `mars_utils.py` no contiene definiciones de modelos
- **Bug conocido en 03_eda_msl.ipynb**: la línea `rec["big_rock_border_px"] = int((np.abs(br_neighbor_y) + np.abs(br_neighbor_x)).sum())` lanza `ValueError` por shapes incompatibles `(255,256)` y `(256,255)`. Corrección: `int(np.abs(br_neighbor_y).sum() + np.abs(br_neighbor_x).sum())`
- **PyTorch mínimo 2.6.0**: versiones anteriores (2.5.x) son incompatibles con `transformers` actual por CVE-2025-32434. Instalar con `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`
- **Patience actualizado a 10**: subido de 7 a 10 para dar más margen de convergencia con el dataset completo
- **Sistema de checkpoints robusto**: `train_model` guarda `_periodic.pth` cada 3 epochs (sobreescribe el mismo archivo) incluyendo el `history` completo acumulado hasta ese punto. `run_multi_seed` guarda `_progress.json` tras cada seed completado, también incluyendo el `history` completo de ese seed. Al reanudar: se salta seeds ya completados, se retoma el seed interrumpido desde el último `_periodic.pth` restaurando el historial previo.
- **`_best.pth` nunca se borra**: es el checkpoint definitivo para evaluación y visualización. Solo se sobreescribe si el nuevo entrenamiento supera el mIoU previo. Si se interrumpe un reentrenamiento accidental, el `_best.pth` original queda intacto mientras no se supere ese mIoU

---

## 16. Bugs Encontrados y Fixes Aplicados (sesión v5)

Todos los bugs encontrados durante la primera ejecución real de `05b_model_segformer.ipynb`.

### Bug 1 — `KeyError: 'train'` en `load_split()`
**Causa**: el pickle `split_indices_msl6k.pkl` usa claves `'train_ids'` y `'val_ids'`, no `'train'` y `'val'` como asumía el código original.
**Fix**: cambiar `.iloc[indices["train"]]` por `.iloc[indices["train_ids"]]` (y equivalente para val). Finalmente se descartó `iloc` completamente — ver Bug 3.

### Bug 2 — `ValueError: invalid literal for int() with base 10: np.str_('NLB_...')`
**Causa**: `train_ids` y `val_ids` en el pickle son **stems de archivo** (strings), no índices posicionales. `.iloc` no puede usarse con strings.
**Fix**: usar filtrado por stem en vez de `.iloc`:
```python
df_full["_stem"] = df_full["image_path"].apply(lambda x: Path(x).stem)
df_train = df_full[df_full["_stem"].isin(train_ids)].drop(columns="_stem").reset_index(drop=True)
df_val   = df_full[df_full["_stem"].isin(val_ids)  ].drop(columns="_stem").reset_index(drop=True)
```

### Bug 3 — `FileNotFoundError` en DataLoader: rutas relativas en los manifests
**Causa**: los manifests (`manifest_msl_train.csv`, `manifest_msl_gold_test.csv`) guardan rutas relativas tipo `msl\ncam\images\edr\...`. El DataLoader las intentaba abrir desde el directorio de trabajo, no desde ROOT. Además, el entrenamiento debe usar las imágenes ya resizadas en `data/images_256/` y `data/masks_256/`, no las originales.
**Fix**: al final de `load_split()`, redirigir rutas para train y val:
```python
for df in [df_train, df_val]:
    df["image_path"] = df["image_path"].apply(
        lambda p: str(DATA_DIR / "images_256" / (Path(p).stem + ".jpg"))
    )
    df["mask_path"] = df["mask_path"].apply(
        lambda p: str(DATA_DIR / "masks_256" / (Path(p).stem + ".png"))
    )
```

### Bug 4 — `FileNotFoundError` en gold set: sufijo `_merged` duplicado
**Causa**: el manifest del gold tiene rutas de máscara con sufijo `_merged` (ej. `NLA_...M1_merged.png`). Al construir la ruta desde el stem de `mask_path` y agregar `_merged` manualmente, quedaba `_merged_merged`.
**Fix**: construir la ruta de máscara del gold desde el stem de `image_path` (que no tiene `_merged`) y agregar `.png` sin sufijo, porque `02_preprocessing.ipynb` ya guardó las máscaras del gold **sin** el sufijo `_merged`:
```python
df_gold["image_path"] = df_gold["image_path"].apply(
    lambda p: str(DATA_DIR / "images_256" / (Path(p).stem + ".jpg"))
)
df_gold["mask_path"] = df_gold["image_path"].apply(
    lambda p: str(DATA_DIR / "masks_256" / (Path(p).stem + ".png"))
)
```

### Bug 5 — `ValueError` en `transformers`: PyTorch < 2.6 incompatible
**Causa**: versión nueva de `transformers` exige PyTorch ≥ 2.6 por CVE-2025-32434.
**Fix**: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`

### Warnings de carga de SegFormer (no son bugs)
Al llamar `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=4)`:
- `UNEXPECTED: classifier.bias / classifier.weight` — cabeza original de ImageNet-1k, se descarta. Normal.
- `MISSING: decode_head.*` — nueva cabeza de 4 clases inicializada aleatoriamente. Normal y esperado.
- `num_labels=4 incompatible with id2label of length 1000` — warning cosmético. No afecta entrenamiento.

### Resultado de validación rápida (fast_subset=True, ~200 imgs train)
```
SegFormer-B2: mIoU=0.7424 ± 0.0185   IC95=±0.0210
IoU soil=0.9592  bedrock=0.9029  sand=0.9218  big_rock=0.1855
Pixel Accuracy=0.9663 | Tiempo=152s/seed | Epoch mejor=17.7
```
Pipeline validado de punta a punta. Listo para `fast_subset=False`.

---

## 17. Bugs Encontrados y Fixes Aplicados (sesión v6 — debug `05c_model_marsseg.ipynb`)

Estos bugs aplican a **todos los notebooks de modelos restantes** (`05a`, `05c`, `05d`, `05e`). El patrón correcto ya está en `05b_model_segformer.ipynb` — usarlo como referencia.

---

### Bug 6 — `KeyError: 'stem'` en la verificación anti-leakage

**Causa**: en los notebooks se intentaba hacer `df_train['stem']` asumiendo que `load_split()` retorna una columna `stem`. Esa columna se crea internamente como `_stem` y se **elimina con `drop(columns="_stem")`** antes de retornar el DataFrame. No existe en el resultado.

**Fix**: derivar el stem directamente de `image_path` en el notebook:
```python
# ✅ CORRECTO — stem derivado de image_path
train_ids = set(Path(p).stem for p in df_train['image_path'])
gold_ids  = set(Path(p).stem for p in df_gold['image_path'])
assert len(train_ids & gold_ids) == 0, '⚠️ DATA LEAKAGE detectado'
```

**Afecta**: `05a`, `05c`, `05d`, `05e` — cualquier notebook que haga `df_train['stem']` o `df_gold['stem']`.

---

### Bug 7 — `TypeError: run_multi_seed() got an unexpected keyword argument 'mean'`

**Causa**: `run_multi_seed()` en `mars_utils.py` llama internamente a `load_norm_stats()` para obtener `mean` y `std`. No los acepta como argumentos externos. Tampoco acepta `max_epochs` — el parámetro correcto se llama `num_epochs`.

**Fix**: quitar `mean`, `std` y cualquier alias incorrecto de la llamada a `run_multi_seed`. La firma correcta es:
```python
summary = run_multi_seed(
    model_fn       = build_model,
    df_train       = df_train,
    df_val         = df_val,
    df_gold        = df_gold,
    criterion_fn   = criterion_fn,
    optimizer_fn   = optimizer_fn,
    scheduler_fn   = scheduler_fn,
    model_name     = MODEL_NAME,
    device         = DEVICE,
    num_epochs     = MAX_EPOCHS,   # ← 'num_epochs', no 'max_epochs'
    patience       = PATIENCE,
    batch_size     = BATCH_SIZE,
    fast_subset    = FAST_SUBSET,
    n_train_fast   = 200,
    n_val_fast     = 50,
)
# mean y std NO van aquí — run_multi_seed los carga solo con load_norm_stats()
```

**Afecta**: `05a`, `05c`, `05d`, `05e`.

> **Diagnóstico rápido**: si hay duda sobre los argumentos exactos, ejecutar `import inspect; print(inspect.signature(run_multi_seed))` antes de la llamada.

---

### Bug 8 — `KeyError: 'best_seed'` al recuperar el mejor checkpoint

**Causa**: el diccionario `summary` devuelto por `run_multi_seed` **no tiene** claves `'best_seed'` ni `'checkpoints'`. Solo contiene: `'per_seed'`, `'mIoU_mean'`, `'mIoU_std'`, `'mIoU_ci95'`, `'iou_per_class'`, `'pixel_acc'`, `'train_time_mean'`, `'best_epoch_mean'`.

**Fix**: derivar `best_seed` y la ruta del checkpoint manualmente, siguiendo el patrón de `05b`:
```python
# ✅ CORRECTO — extraer best_seed de summary['per_seed']
best_seed = max(summary["per_seed"], key=lambda r: r["mIoU"])["seed"]
ckpt_path = CHECKPOINTS_DIR / f"{MODEL_NAME}_seed{best_seed}_best.pth"

best_model = build_model().to(DEVICE)
ckpt = torch.load(ckpt_path, map_location=DEVICE)
best_model.load_state_dict(ckpt["model_state"])
best_model.eval()

print(f"Mejor seed: {best_seed} | mIoU gold: {max(summary['per_seed'], key=lambda r: r['mIoU'])['mIoU']:.4f}")

visualize_predictions(best_model, df_gold, DEVICE, mean=mean, std=std, n=5)
```

**Afecta**: `05a`, `05c`, `05d`, `05e` — la sección de visualización cualitativa de todos los notebooks.

> **Nota**: `CHECKPOINTS_DIR` ya viene importado desde `mars_utils` con `from mars_utils import *`. No redefinirlo en el notebook.

---

### Bug 9 — Duplicación de filas en `benchmark_results.csv` al reejecutar

**Causa**: `append_benchmark_results()` abre el CSV en modo `"a"` (append). Si se corre el notebook dos veces, aparecen dos filas con el mismo modelo.

**Fix**: limpiar la fila del modelo actual antes de llamar a `append_benchmark_results()`:
```python
import pandas as pd
from mars_utils import BENCHMARK_CSV

# Eliminar fila anterior del mismo modelo si existe
if BENCHMARK_CSV.exists():
    df_csv = pd.read_csv(BENCHMARK_CSV)
    df_csv = df_csv[df_csv["model"] != MODEL_NAME]
    df_csv.to_csv(BENCHMARK_CSV, index=False)

append_benchmark_results(summary, params_M=params_M)
```

**Afecta**: todos los notebooks de modelos. Especialmente importante si se reejecutar después de un `fast_subset=True` para no mezclar resultados parciales con los completos.

---

### Bug 10 — Gráficas de entrenamiento incompletas al reanudar / seed incorrecto elegido como mejor

**Causa**: `train_model` inicializaba `history = {"train": [], "val": []}` siempre desde cero, sin restaurarlo del checkpoint periódico. Al reanudar, las epochs previas al corte se perdían y la gráfica empezaba desde "epoch 1" aunque fuera la epoch 47. Adicionalmente, `_progress.json` guardaba los resultados de seeds completados **sin** su `history`, por lo que al reanudar en el último seed, `plot_best_seed_curves` no podía comparar los 3 seeds y elegía el único con history en memoria (el seed en curso), independientemente de su mIoU real.

**Fix aplicado en `mars_utils.py` (v8)**:

1. `train_model` restaura `history` desde el `_periodic.pth` al reanudar:
```python
history = ckpt.get("history", {"train": [], "val": []})
```

2. El checkpoint periódico ahora incluye `history` y se guarda cada **3 epochs** (antes 5):
```python
if epoch % 3 == 0:
    torch.save({..., "history": history}, periodic_ckpt)
```

3. Se añadió `_serialize_history()` — convierte el history a tipos nativos Python (float, dict) para serialización JSON, manejando `numpy.float32/64`, tensors escalares y dicts de IoU por clase.

4. `_progress.json` ahora incluye el history completo de cada seed completado:
```python
{"history": _serialize_history(r["history"])}
```

**Ciclo de vida de archivos tras el fix**:

| Archivo | Contiene history | Se destruye |
|---------|-----------------|-------------|
| `{model}_seed{N}_periodic.pth` | Sí — history hasta epoch actual | Al terminar ese seed |
| `{model}_progress.json` | Sí — history completo de seeds terminados | Al terminar los 3 seeds |
| `{model}_seed{N}_best.pth` | No | Nunca |

**Compatibilidad hacia atrás**: checkpoints periódicos generados antes del fix no tienen `"history"`. El `.get(..., {"train": [], "val": []})` activa el fallback y se comporta igual que antes — no peor.

**Afecta**: `mars_utils.py` — todos los notebooks que usen `run_multi_seed` se benefician automáticamente sin cambios en los notebooks.

---

### Plantilla estándar para notebooks `05a`, `05c`, `05d`, `05e`

Resumen del bloque de datos + verificación anti-leakage que debe ir al inicio de cada notebook (después del import y carga de split):

```python
# ── Carga de datos ──────────────────────────────────────────────────────────
df_train, df_val, df_gold = load_split()
mean, std = load_norm_stats()

# ── Verificación anti-leakage ────────────────────────────────────────────────
train_ids = set(Path(p).stem for p in df_train['image_path'])
gold_ids  = set(Path(p).stem for p in df_gold['image_path'])
assert len(train_ids & gold_ids) == 0, '⚠️ DATA LEAKAGE detectado'
print(f'✅ Train: {len(df_train)} | Val: {len(df_val)} | Gold: {len(df_gold)}')
print(f'Normalización — mean: {mean} | std: {std}')
```

---

## 15. Instrucciones para Compañeros (onboarding)

Ver `README.md` en la raiz del repositorio para las instrucciones completas.

Resumen de los puntos criticos:

1. **Clonar el repositorio** desde `https://github.com/UnDauphin/ai4mars_DL-v3` via GitHub Desktop
2. **NO descargar datos desde Zenodo** — `data/images_256/` y `data/masks_256/` ya estan en el repo (~137 MB)
3. **NO ejecutar** los notebooks `01` ni `02` — todo el preprocesamiento ya esta hecho
4. **Crear el entorno** siguiendo la seccion 4 de este documento — usar PyTorch 2.6.0, no 2.5.x
5. **Crear su branch** en GitHub Desktop antes de tocar cualquier archivo: `resultados-nombremodelo`
6. **NO regenerar** `processed/split_indices_msl6k.pkl` — usar el que viene en el repo
7. **Verificar `FAST_SUBSET`** — si está en `True`, cambiarlo a `False` antes de ejecutar. Si ya está en `False`, no tocarlo
8. **Al terminar**: commitear el notebook ejecutado (con graficas visibles) y el CSV via GitHub Desktop, luego abrir Pull Request
9. **Si el entrenamiento se interrumpe**: volver a ejecutar la celda de `run_multi_seed` — el codigo detecta automaticamente el `_progress.json` y los `_periodic.pth` y reanuda desde donde se quedo

## 18. Distribucion del Proyecto — Repositorio Git y Flujo de Trabajo

### Repositorio

URL: `https://github.com/UnDauphin/ai4mars_DL-v3`

### Que esta en el repositorio

Los datos resizeados (`data/images_256/` y `data/masks_256/`) se suben al repositorio porque pesan ~137 MB en total (12.644 archivos JPEG/PNG de 256x256, ~20-50 KB cada uno), dentro del limite practico de GitHub. Esto elimina la necesidad de que los companeros descarguen y ejecuten los notebooks de preprocesamiento.

### .gitignore final

```gitignore
# Datos originales (demasiado grandes para git)
msl/
mer/
m2020/
# data/ <- NO se ignora — las imagenes resizeadas si van al repo

# Checkpoints (archivos de varios cientos de MB por modelo)
checkpoints/

# Processed: ignorar todo EXCEPTO artefactos necesarios para reproducibilidad
processed/*
!processed/split_indices_msl6k.pkl
!processed/normalization_stats.json
!processed/class_weights.json
!processed/manifest_msl_train.csv
!processed/manifest_msl_gold_test.csv

# Entornos Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Archivos del sistema
.DS_Store
Thumbs.db
desktop.ini

# IDEs
.vscode/
.idea/
```

Nota: `results/` ya NO esta en `.gitignore` — el `benchmark_results.csv` se sube al repo via Pull Request de cada companero.

### Flujo de trabajo con companeros (branches)

Cada companero trabaja en su propia branch para evitar conflictos en `benchmark_results.csv`:

1. Crear branch `resultados-nombremodelo` desde GitHub Desktop antes de empezar
2. Al terminar el entrenamiento, commitear en su branch:
   - El notebook ejecutado con todas las celdas y graficas visibles
   - `results/benchmark_results.csv` con la fila del modelo
3. Abrir Pull Request hacia `main` en GitHub.com
4. El responsable principal revisa y aprueba el merge
5. Al hacer merge, resolver el conflicto en el CSV manteniendo todas las filas — cada PR agrega una fila nueva

### Configuracion de Colab Pro para 05a (DeepLabV3+)

El notebook `05a` usa Google Colab Pro (A100/T4). La celda de setup:

```python
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    !git clone https://github.com/UnDauphin/ai4mars_DL-v3 /content/ai4mars_DL-v3
    !pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 -q
    !pip install -r /content/ai4mars_DL-v3/requirements.txt -q
    PROJECT_ROOT = Path('/content/ai4mars_DL-v3')
    sys.path.append(str(PROJECT_ROOT / 'notebooks'))
    import mars_utils
    mars_utils.CHECKPOINTS_DIR = Path('/content/drive/MyDrive/ai4mars_DL-v3/checkpoints')
    mars_utils.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print('Colab — Drive montado, repo clonado, dependencias instaladas')
```

Drive se monta para persistir los checkpoints (`_best.pth`) en caso de corte de sesion. El repo se clona para tener el codigo y los datos. Los checkpoints se redirigen a Drive sobreescribiendo `mars_utils.CHECKPOINTS_DIR` despues del import.

### Lo que se guarda tras el entrenamiento y donde

| Archivo | Ubicacion | Persistencia | En el repo |
|---------|-----------|-------------|------------|
| `{modelo}_seed{N}_best.pth` | `checkpoints/` (local) o Drive (Colab) | Permanente | No (.gitignore) |
| `{modelo}_seed{N}_periodic.pth` | `checkpoints/` | Temporal (se sobreescribe cada 5 epochs) | No |
| `{modelo}_progress.json` | `checkpoints/` | Temporal (se borra al completar) | No |
| `benchmark_results.csv` | `results/` | Permanente | Si (via PR) |
| Notebook ejecutado con graficas | `notebooks/` | Permanente | Si (via PR) |

Los checkpoints pesan ~170 MB por modelo en DeepLabV3+, menos en los otros. Se comparten via Google Drive si son necesarios para visualizaciones posteriores.

---

*Documento actualizado — v8. Reemplaza completamente la v7.*
*Últimos cambios: Bug 10 documentado (history perdido al reanudar + seed incorrecto como mejor); `mars_utils.py` actualizado con Fix 1-3 y `_serialize_history()`; intervalo de checkpoint periódico cambiado de 5 a 3 epochs; instrucciones para compañeros actualizadas (punto 7 reescrito); sección 14 actualizada con descripción del sistema de checkpoints.*
*Próximos pasos: lanzar entrenamientos completos en paralelo — 05a en Colab Pro, 05b/05c en laptop principal, 05d en compañero 1, 05e en compañero 2.*
