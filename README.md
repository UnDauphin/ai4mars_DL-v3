# ai4mars_DL-v3
## Segmentacion Semantica de Terreno Marciano — Entregable 2 Deep Learning

Benchmark de modelos de segmentacion semantica sobre el dataset AI4MARS (NASA/JPL).
El objetivo es clasificar a nivel de pixel el tipo de terreno en imagenes capturadas por rovers en la superficie de Marte.

---

## Clases

| ID | Clase | Nota |
|----|-------|------|
| 0 | Soil (Suelo) | Clase mayoritaria |
| 1 | Bedrock (Roca base) | Clase mayoritaria |
| 2 | Sand (Arena) | Clase media |
| 3 | Big Rock (Rocas sueltas) | Clase minoritaria (~2% de pixeles) |
| 255 | Ignore | Pixeles sin etiquetar — excluidos de loss y metricas |

---

## Modelos implementados

| Notebook | Modelo | Backbone | Maquina |
|----------|--------|----------|---------|
| 05a | DeepLabV3+ | ResNet-50 | Google Colab Pro |
| 05b | IC-TransUNet-AI4MARS | InceptionNeXt + CSWin dual branch | Google Colab Pro / recuperacion checkpoints |
| 05f | SegFormer-B2 | MiT-B2 | Laptop principal |
| 05c | MarsSeg | ResNet-50 + MiniASPP + PSA + SPPM | Laptop principal |
| 05d | TerSeg | ResNet-34 + Swin-Tiny | Companero 1 |
| 05e | DepthFormer-RGB | Swin-Tiny + UPerNet + PPM | Companero 2 |

---

## Estructura del repositorio

```
ai4mars_DL-v3/
├── data/
│   ├── images_256/        — 6.322 imagenes MSL resizeadas a 256x256 (JPEG)
│   └── masks_256/         — 6.322 mascaras correspondientes (PNG)
├── processed/
│   ├── split_indices_msl6k.pkl       — split fijo train/val (NO regenerar)
│   ├── normalization_stats.json      — media y std del train set
│   ├── class_weights.json            — pesos de clase para la loss
│   ├── manifest_msl_train.csv        — manifest del pool de entrenamiento
│   └── manifest_msl_gold_test.csv    — manifest del gold test set (322 imgs)
├── notebooks/
│   ├── mars_utils.py                 — infraestructura comun (datasets, training, metricas)
│   ├── 01_eda_exploratorio.ipynb     — EDA general (ya ejecutado)
│   ├── 02_preprocessing.ipynb        — preprocesamiento (ya ejecutado)
│   ├── 03_eda_msl.ipynb              — EDA especifico MSL (ya ejecutado)
│   ├── 04_marco_teorico.md           — revision de literatura
│   ├── 05a_model_deeplabv3plus.ipynb
│   ├── 05b_model_transunet.ipynb
│   ├── 05c_model_marsseg.ipynb
│   ├── 05d_model_terseg.ipynb
│   ├── 05e_model_depthformer.ipynb
│   └── 05f_model_segformer.ipynb
├── results/
│   └── benchmark_results.csv         — metricas agregadas de todos los modelos
├── requirements.txt
└── .gitignore
```

---

## Instrucciones para companeros

### Requisitos previos

- NVIDIA Driver actualizado
- Anaconda o Miniconda instalado
- GitHub Desktop instalado
- VS Code con la extension de Jupyter

### 1. Clonar el repositorio

Abrir GitHub Desktop, ir a **File > Clone repository**, pegar la URL:

```
https://github.com/UnDauphin/ai4mars_DL-v3
```

Elegir la carpeta donde quieres que quede el proyecto y clonar.

### 2. Crear tu branch

Antes de tocar cualquier archivo, crear una branch con tu nombre desde GitHub Desktop:

- En la barra superior donde dice `main`, hacer click
- Click en **New branch**
- Nombrarla `resultados-nombremodelo` (ejemplo: `resultados-terseg`)
- Click en **Create branch**

Trabajar siempre en tu branch, nunca en `main`.

### 3. Crear el entorno

Abrir Anaconda Prompt (o terminal) y ejecutar:

```powershell
cd ruta/a/ai4mars_DL-v3

conda create -n mars_dl python=3.11 -y
conda activate mars_dl

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python -m ipykernel install --user --name mars_dl --display-name "Mars DL"
```

### 4. Verificar que todo esta bien

Abrir Python y ejecutar:

```python
import torch
print(torch.__version__)           # debe decir 2.6.0+cu124
print(torch.cuda.is_available())   # debe decir True
```

### 5. Ejecutar el notebook

1. Abrir VS Code en la carpeta del proyecto
2. Abrir el notebook asignado (`05d` o `05e` segun corresponda)
3. Seleccionar el kernel **Mars DL** en la esquina superior derecha
4. Ejecutar todas las celdas en orden con **Run All**
5. El entrenamiento completo tarda entre 15 y 20 horas — se puede dejar corriendo

> El parametro `FAST_SUBSET` ya esta en `False`. No lo cambies.

### 6. Al terminar — subir resultados

Una vez que el notebook termino de ejecutarse completamente (incluyendo las visualizaciones de la seccion 7):

1. En GitHub Desktop verificar que estas en tu branch (no en `main`)
2. Van a aparecer dos archivos modificados:
   - `notebooks/05d_model_terseg.ipynb` (o el que corresponda) — el notebook con las celdas ejecutadas y las graficas
   - `results/benchmark_results.csv` — las metricas del modelo
3. Escribir un mensaje de commit descriptivo, por ejemplo: `resultados: TerSeg entrenamiento completo`
4. Click en **Commit to resultados-nombremodelo**
5. Click en **Push origin**
6. En GitHub.com abrir un **Pull Request** desde tu branch hacia `main`
7. Avisar por el grupo para que se apruebe el merge

### Notas importantes

- **No ejecutar** los notebooks `01` ni `02` — los datos ya estan procesados
- **No regenerar** `processed/split_indices_msl6k.pkl` — todos los modelos deben usar exactamente el mismo split
- Los checkpoints (archivos `.pth`) se guardan en `checkpoints/` — esa carpeta esta en `.gitignore`, no hace falta subirla
- Si el entrenamiento se interrumpe, volver a ejecutar la celda de `run_multi_seed` — el codigo detecta automaticamente el progreso guardado y retoma desde donde se quedo

---

## Notas para Google Colab (notebook 05a)

El notebook `05a_model_deeplabv3plus.ipynb` esta configurado para correr en Google Colab Pro. Al ejecutar la primera celda se monta Drive automaticamente, se clona el repo y se instalan las dependencias. Los checkpoints se guardan en Drive para sobrevivir cortes de sesion.

---

## Entorno de desarrollo

| Paquete | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |

Ver `requirements.txt` para la lista completa de dependencias.
