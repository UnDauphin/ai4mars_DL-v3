# Segmentacion semantica de terreno marciano

**Entregable 3 - Proyecto de Deep Learning**

## Integrantes del grupo

| Nombre | Codigo |
|---|---|
| Alejandro David Moya Nieves | 200196366 |
| Mateo Jose Gomez Rojas | 200193297 |
| Mateo Andres Molinares Tellez | 200196955 |
| David Alejandro Ibanez Barrios | 200195861 |

Este Jupyter Book presenta un sistema de segmentacion semantica de terreno marciano sobre el dataset AI4MARS de NASA/JPL. El objetivo es clasificar cada pixel de imagenes NavCam del rover Curiosity en cuatro clases de terreno: `soil`, `bedrock`, `sand` y `big_rock`, ignorando los pixeles sin etiqueta valida (`255`).

El problema es relevante para navegacion autonoma de rovers y analisis geologico: un rover necesita distinguir suelo firme, arena suelta, roca base y rocas individuales para planear rutas seguras sin depender de control humano en tiempo real. AI4MARS es especialmente dificil porque combina imagenes de bajo contraste, clases visualmente parecidas, etiquetas con incertidumbre por crowdsourcing, redimensionamiento a 256x256 px y un desbalance extremo donde `big_rock` representa menos del 1% de los pixeles.

## Propuesta original

La propuesta original del proyecto es la adaptacion y evaluacion de **SegFormer-B2** al dominio AI4MARS. La originalidad no consiste en inventar una arquitectura desde cero, sino en aplicar una arquitectura Transformer jerarquica de referencia a un dominio planetario no cubierto por la revision bibliografica del proyecto, con un protocolo reproducible y comparacion contra benchmarks fuertes.

SegFormer-B2 se defiende como modelo central por su encoder multiescala MiT-B2, su decoder MLP ligero, su robustez a cambios de resolucion y su buen balance entre rendimiento, estabilidad y complejidad. En los resultados del proyecto obtiene la mayor pixel accuracy, la menor varianza entre seeds y resultados competitivos en mIoU frente a modelos mas pesados.

## Pipeline experimental

El proyecto sigue un pipeline reproducible:

- Analisis exploratorio general del dataset y sus misiones.
- Seleccion justificada del subconjunto MSL Curiosity NavCam.
- Preprocesamiento comun a 256x256 px, normalizacion calculada solo con train y split fijo.
- Revision de literatura para elegir la propuesta original y los modelos benchmark.
- Entrenamiento multi-seed con evaluacion en un gold test set fijo.
- Benchmark final con metricas por clase, intervalos de confianza y pruebas estadisticas.
- Discusion critica de discrepancias visuales entre mascaras predichas y ground truth.

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

## Modelos evaluados

| Rol | Modelo | Familia | Notebook |
|---|---|---|---|
| Propuesta original | SegFormer-B2 | Transformer jerarquico | `05f_model_segformer.ipynb` |
| Benchmark | DeepLabV3+ | CNN + ASPP | `05a_model_deeplabv3plus.ipynb` |
| Benchmark | MarsSeg | CNN avanzado para terreno marciano | `05c_model_marsseg.ipynb` |
| Benchmark | TerSeg | Hibrido CNN + Transformer | `05d_model_terseg.ipynb` |
| Benchmark | DepthFormer-RGB | Swin Transformer + UPerNet | `05e_model_depthformer.ipynb` |
| Benchmark | IC-TransUNet-AI4MARS | CNN + Transformer dual branch | `05b_model_transunet.ipynb` |

La metrica principal es mIoU, complementada con pixel accuracy, IoU por clase, tiempo de entrenamiento, numero de parametros y pruebas estadisticas. La lectura final prioriza el balance entre precision, estabilidad y costo computacional, no una unica cifra aislada.
