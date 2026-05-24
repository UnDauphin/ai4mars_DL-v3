# Segmentacion semantica de terreno marciano

**Entregable 2 - Proyecto de Deep Learning**

## Integrantes del grupo

| Nombre | Codigo |
|---|---|
| Alejandro David Moya Nieves | 200196366 |
| Mateo Jose Gomez Rojas | 200193297 |
| Mateo Andres Molinares Tellez | 200196955 |
| David Alejandro Ibanez Barrios | 200195861 |

Este Jupyter Book presenta el entregable de Deep Learning para segmentacion semantica de terreno marciano sobre el dataset AI4MARS de NASA/JPL. El objetivo es comparar modelos de referencia capaces de clasificar cada pixel de imagenes de rovers en cuatro clases de terreno: `soil`, `bedrock`, `sand` y `big_rock`, ignorando pixeles sin etiqueta (`255`).

El proyecto sigue un pipeline reproducible:

- Analisis exploratorio general del dataset y sus misiones.
- Seleccion justificada del subconjunto MSL Curiosity NavCam.
- Preprocesamiento comun a 256x256 px, normalizacion calculada solo con train y split fijo.
- Revision de literatura y seleccion de cinco modelos benchmark.
- Entrenamiento multi-seed con evaluacion en un gold test set fijo.
- Benchmark final con metricas por clase, intervalos de confianza y pruebas estadisticas.

## Diseno experimental

Todos los modelos se comparan bajo las mismas condiciones:

| Elemento | Configuracion |
|---|---|
| Dataset de entrenamiento | AI4MARS MSL Curiosity NavCam |
| Split | 70% train / 30% validation sobre subset MSL 6k |
| Test | Gold set MSL min3-100agree, 322 imagenes |
| Resolucion | 256x256 px |
| Clases | soil, bedrock, sand, big_rock |
| Ignore index | 255 |
| Seeds | 42, 123, 7 |
| Metrica principal | mean IoU sobre gold test |

## Modelos benchmark

| Modelo | Familia | Notebook |
|---|---|---|
| DeepLabV3+ | CNN + ASPP | `05a_model_deeplabv3plus.ipynb` |
| SegFormer-B2 | Transformer | `05b_model_segformer.ipynb` |
| MarsSeg | CNN avanzado para terreno marciano | `05c_model_marsseg.ipynb` |
| TerSeg | Hibrido CNN + Transformer | `05d_model_terseg.ipynb` |
| DepthFormer-RGB | Swin Transformer + UPerNet | `05e_model_depthformer.ipynb` |

La metrica principal es mIoU, complementada con pixel accuracy, IoU por clase, tiempo de entrenamiento, numero de parametros y pruebas estadisticas para comparar diferencias entre modelos.
