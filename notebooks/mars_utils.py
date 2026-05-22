"""
mars_utils.py  —  v3
====================
Infraestructura compartida para los notebooks de modelos benchmark (05a–05e).
Compatible con la estructura de directorios v3 del proyecto.

Ubicación en el proyecto:
    notebooks/mars_utils.py        ← este archivo
    notebooks/05a_model_*.ipynb    ← importan con: from mars_utils import *

Rutas clave (todas relativas a ROOT = carpeta raíz del proyecto):
    ROOT/processed/split_indices_msl6k.pkl
    ROOT/processed/normalization_stats.json
    ROOT/processed/class_weights.json
    ROOT/processed/manifest_msl_gold_test.csv
    ROOT/data/images_256/
    ROOT/data/masks_256/
    ROOT/checkpoints/
    ROOT/results/benchmark_results.csv
"""

import os, json, random, pickle, time, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS  —  ROOT es la carpeta raíz del proyecto (parent de notebooks/)
# ─────────────────────────────────────────────────────────────────────────────

def _find_root() -> Path:
    """
    Encuentra ROOT (carpeta raíz del proyecto) de forma robusta,
    funciona tanto en local como en Google Colab con Drive montado.
    """
    # En local: ROOT es parent.parent de este archivo
    local = Path(__file__).resolve().parent.parent
    if (local / "processed").exists():
        return local
    # En Colab: buscar desde el directorio de trabajo actual
    cwd = Path(os.getcwd())
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "processed").exists():
            return candidate
    raise RuntimeError(
        "No se encontró ROOT del proyecto. "
        "Verifica que 'processed/' existe en la raíz y que el path es correcto."
    )

ROOT = _find_root()

PROCESSED_DIR = ROOT / "processed"
DATA_DIR      = ROOT / "data"
IMAGES_256    = DATA_DIR / "images_256"
MASKS_256     = DATA_DIR / "masks_256"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"

for _d in [CHECKPOINTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SPLIT_PATH      = PROCESSED_DIR / "split_indices_msl6k.pkl"
NORM_STATS_PATH = PROCESSED_DIR / "normalization_stats.json"
CLASS_W_PATH    = PROCESSED_DIR / "class_weights.json"
GOLD_MANIFEST   = PROCESSED_DIR / "manifest_msl_gold_test.csv"
BENCHMARK_CSV   = RESULTS_DIR   / "benchmark_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 4
IGNORE_INDEX = 255
IMG_SIZE     = 256
BATCH_SIZE   = 16
SEEDS        = [42, 123, 7]

CLASS_NAMES  = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
CLASS_RGB    = {
    0:   (230/255, 159/255,   0/255),   # soil      — naranja
    1:   ( 86/255, 180/255, 233/255),   # bedrock   — azul claro
    2:   (  0/255, 158/255, 115/255),   # sand      — verde
    3:   (213/255,  94/255,   0/255),   # big_rock  — rojo
    255: (0.6,     0.6,      0.6   ),   # ignore    — gris
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN  —  cargada desde normalization_stats.json (calculada en train)
# ─────────────────────────────────────────────────────────────────────────────
def load_norm_stats():
    """
    Carga mean y std calculados sobre el train set propio.
    Falla explícitamente si el archivo no existe (ejecutar 02_preprocessing.ipynb primero).
    """
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {NORM_STATS_PATH}. "
            "Ejecuta 02_preprocessing.ipynb antes de usar mars_utils."
        )
    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)
    mean = stats["mean"]   # lista de 3 floats (canal R, G, B)
    std  = stats["std"]    # lista de 3 floats
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# SEMILLA
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS  —  sin resize (imágenes ya están en 256×256)
# ─────────────────────────────────────────────────────────────────────────────
class JointTransformTrain:
    """
    Augmentaciones para train.
    SIN resize — las imágenes ya están en 256×256 desde 02_preprocessing.ipynb.
    Normalización con stats calculadas sobre el train set (no ImageNet).
    """
    def __init__(self, mean, std):
        self.color_jit = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, img: Image.Image, mask: Image.Image):
        # Flip horizontal
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Flip vertical
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        # Rotación ±15° — máscara rellena con IGNORE_INDEX, nunca con clase válida
        angle = random.uniform(-15, 15)
        img   = TF.rotate(img,  angle,
                          interpolation=T.InterpolationMode.BILINEAR, fill=0)
        mask  = TF.rotate(mask, angle,
                          interpolation=T.InterpolationMode.NEAREST,  fill=IGNORE_INDEX)
        # Color jitter (solo imagen)
        img = self.color_jit(img)
        # ToTensor + Normalize
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


class JointTransformVal:
    """
    Solo ToTensor + Normalize. Sin augmentaciones.
    Usada para val, test y visualización.
    """
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean=mean, std=std)
        self.mean = np.array(mean)
        self.std  = np.array(std)

    def __call__(self, img: Image.Image, mask: Image.Image):
        img  = self.normalize(TF.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class MarsTerrainDataset(Dataset):
    """
    Lee imágenes y máscaras desde data/images_256/ y data/masks_256/.
    Espera un DataFrame con columnas: image_path, mask_path.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL SPLIT  —  solo lectura, nunca regenerar
# ─────────────────────────────────────────────────────────────────────────────
def load_split(manifest_train_path: Path = PROCESSED_DIR / "manifest_msl_train.csv",
               split_path: Path = SPLIT_PATH):
    if not split_path.exists():
        raise FileNotFoundError(
            f"No se encontró {split_path}. "
            "Verifica que ejecutaste 02_preprocessing.ipynb y que el archivo "
            "está en ROOT/processed/."
        )
    if not GOLD_MANIFEST.exists():
        raise FileNotFoundError(
            f"No se encontró {GOLD_MANIFEST}. "
            "Verifica que ejecutaste 01_eda_exploratorio.ipynb."
        )

    df_full = pd.read_csv(manifest_train_path)

    with open(split_path, "rb") as f:
        indices = pickle.load(f)

    train_ids = set(indices["train_ids"])
    val_ids   = set(indices["val_ids"])

    df_full["_stem"] = df_full["image_path"].apply(lambda x: Path(x).stem)

    df_train = df_full[df_full["_stem"].isin(train_ids)].drop(columns="_stem").reset_index(drop=True)
    df_val   = df_full[df_full["_stem"].isin(val_ids)  ].drop(columns="_stem").reset_index(drop=True)
    df_gold  = pd.read_csv(GOLD_MANIFEST)

    # Verificación anti data-leakage
    _train_ids = set(df_train["image_path"].apply(lambda x: Path(x).stem))
    _gold_ids  = set(df_gold["image_path"].apply(lambda x: Path(x).stem))
    assert len(_train_ids & _gold_ids) == 0, \
        "DATA LEAKAGE: hay solapamiento entre train y gold test set."

    # Redirigir train y val
    for df in [df_train, df_val]:
        df["image_path"] = df["image_path"].apply(
            lambda p: str(DATA_DIR / "images_256" / (Path(p).stem + ".jpg"))
        )
        df["mask_path"] = df["mask_path"].apply(
            lambda p: str(DATA_DIR / "masks_256" / (Path(p).stem + ".png"))
        )

    # Redirigir gold — máscara tiene sufijo _merged
    df_gold["image_path"] = df_gold["image_path"].apply(
        lambda p: str(DATA_DIR / "images_256" / (Path(p).stem + ".jpg"))
    )
    df_gold["mask_path"] = df_gold["image_path"].apply(
        lambda p: str(DATA_DIR / "masks_256" / (Path(p).stem + ".png"))
    )

    
    print(f"✅ Split cargado — train: {len(df_train)} | val: {len(df_val)} | gold test: {len(df_gold)}")
    return df_train, df_val, df_gold


# ─────────────────────────────────────────────────────────────────────────────
# SUBSET RÁPIDO  —  para pruebas en tu PC antes de pasar a compañeros
# ─────────────────────────────────────────────────────────────────────────────
def make_fast_subset(df_train: pd.DataFrame, df_val: pd.DataFrame,
                     n_train: int = 200, n_val: int = 50,
                     seed: int = 42) -> tuple:
    """
    Devuelve un subset pequeño estratificado para pruebas rápidas.

    Uso típico en el notebook antes de pasar el código a compañeros:
        df_tr, df_va = make_fast_subset(df_train, df_val, n_train=200, n_val=50)

    Para el entrenamiento real, usar df_train y df_val completos.

    Parameters
    ----------
    n_train : int
        Número de imágenes del subset de train (default 200 → ~1 min/epoch en RTX 4050).
    n_val   : int
        Número de imágenes del subset de val.
    """
    rng = np.random.default_rng(seed)

    def _stratified_sample(df, n):
        n = min(n, len(df))
        # Intentar estratificar por dominant_class si existe, si no muestrear directo
        if "dominant_class" in df.columns:
            groups = df.groupby("dominant_class", group_keys=False)
            per_group = max(1, n // df["dominant_class"].nunique())
            sampled = groups.apply(
                lambda x: x.sample(min(len(x), per_group), random_state=seed)
            )
            if len(sampled) < n:
                remaining = df.drop(sampled.index)
                extra = remaining.sample(min(n - len(sampled), len(remaining)),
                                         random_state=seed)
                sampled = pd.concat([sampled, extra])
            return sampled.sample(min(n, len(sampled)),
                                   random_state=seed).reset_index(drop=True)
        else:
            return df.sample(n, random_state=seed).reset_index(drop=True)

    sub_train = _stratified_sample(df_train, n_train)
    sub_val   = _stratified_sample(df_val,   n_val)

    print(f"⚡ Subset rápido — train: {len(sub_train)} | val: {len(sub_val)}")
    print("   (Para entrenamiento real usar df_train y df_val completos)")
    return sub_train, sub_val


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(df_train: pd.DataFrame,
                      df_val:   pd.DataFrame,
                      df_gold:  pd.DataFrame,
                      mean, std,
                      batch_size:  int = BATCH_SIZE,
                      num_workers: int = 4) -> tuple:
    """
    Construye los tres DataLoaders: train (con augmentaciones), val y gold test.

    Parameters
    ----------
    mean, std    : stats de normalización cargadas con load_norm_stats()
    num_workers  : 4 funciona bien en Windows con 12 hilos lógicos.
                   Bajar a 0 si hay problemas de multiprocessing en debug.
    """
    pw = num_workers > 0   # persistent_workers requiere num_workers > 0

    train_ds = MarsTerrainDataset(df_train, JointTransformTrain(mean, std))
    val_ds   = MarsTerrainDataset(df_val,   JointTransformVal(mean, std))
    gold_ds  = MarsTerrainDataset(df_gold,  JointTransformVal(mean, std))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=pw
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=pw
    )
    gold_loader = DataLoader(
        gold_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=pw
    )
    return train_loader, val_loader, gold_loader


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
class MetricsAccumulator:
    """
    Acumula TP/FP/FN por clase durante un epoch completo.
    Ignora píxeles con valor IGNORE_INDEX (255).
    """
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp      = np.zeros(self.num_classes, dtype=np.int64)
        self.fp      = np.zeros(self.num_classes, dtype=np.int64)
        self.fn      = np.zeros(self.num_classes, dtype=np.int64)
        self.correct = 0
        self.total   = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds   : logits [B, C, H, W] o predicciones [B, H, W]
        targets : máscaras [B, H, W] con valores {0,1,2,3,255}
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        valid = t != self.ignore_index
        p, t  = p[valid], t[valid]

        self.correct += int((p == t).sum())
        self.total   += int(valid.sum())
        for c in range(self.num_classes):
            self.tp[c] += int(((p == c) & (t == c)).sum())
            self.fp[c] += int(((p == c) & (t != c)).sum())
            self.fn[c] += int(((p != c) & (t == c)).sum())

    def compute(self) -> dict:
        iou = {}
        for c in range(self.num_classes):
            denom = self.tp[c] + self.fp[c] + self.fn[c]
            iou[CLASS_NAMES[c]] = float(self.tp[c] / denom) if denom > 0 else 0.0
        return {
            "mIoU":           float(np.mean(list(iou.values()))),
            "IoU_per_class":  iou,
            "pixel_accuracy": self.correct / max(self.total, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        dice  = 0.0
        n     = 0
        for c in range(self.num_classes):
            t = (targets[valid] == c).float()
            if t.sum() == 0:
                continue
            p     = probs[:, c][valid]
            inter = (p * t).sum()
            dice += (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            n    += 1
        return 1.0 - dice / max(n, 1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.ce           = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                                reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = self.ce(logits, targets)
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss[targets != self.ignore_index].mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  —  AMP con scaler creado UNA sola vez fuera del loop de epochs
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, aux_weight=0.0) -> dict:
    """
    Un epoch de entrenamiento. El scaler se crea fuera (en train_model) y
    se reutiliza para todo el entrenamiento — no recrear por epoch.
    """
    model.train()
    total_loss = 0.0
    acc        = MetricsAccumulator()

    for imgs, masks in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(imgs)
                logits = output["out"] if isinstance(output, dict) else output
                loss   = criterion(logits, masks)
                if aux_weight > 0 and isinstance(output, dict) and "aux" in output:
                    loss = loss + aux_weight * criterion(output["aux"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(imgs)
            logits = output["out"] if isinstance(output, dict) else output
            loss   = criterion(logits, masks)
            if aux_weight > 0 and isinstance(output, dict) and "aux" in output:
                loss = loss + aux_weight * criterion(output["aux"], masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        acc.update(logits.detach(), masks)

    r = acc.compute()
    r["loss"] = total_loss / len(loader)
    return r


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None) -> dict:
    model.eval()
    total_loss = 0.0
    acc        = MetricsAccumulator()

    for imgs, masks in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(imgs)
                logits = output["out"] if isinstance(output, dict) else output
                loss   = criterion(logits, masks)
        else:
            output = model(imgs)
            logits = output["out"] if isinstance(output, dict) else output
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        acc.update(logits, masks)

    r = acc.compute()
    r["loss"] = total_loss / len(loader)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL  —  loop completo con early stopping y checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader,
                optimizer, scheduler, criterion, device,
                model_name: str = "model",
                num_epochs:  int = 80,
                patience:    int = 10,
                aux_weight: float = 0.0,
                use_amp:    bool = True) -> tuple:
    use_amp = use_amp and (device != "cpu") and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    best_state  = None
    best_miou   = -1.0
    best_epoch  = 1
    no_improve  = 0
    history     = {"train": [], "val": []}
    start_epoch = 1
    t_start     = time.time()

    # ── Reanudar desde checkpoint periódico si existe ────────────────────────
    periodic_ckpt = CHECKPOINTS_DIR / f"{model_name}_periodic.pth"
    if periodic_ckpt.exists():
        print(f"Checkpoint periódico encontrado — reanudando desde {periodic_ckpt.name}")
        ckpt = torch.load(periodic_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt["best_miou"]
        no_improve  = ckpt["no_improve"]
        history     = ckpt.get("history", {"train": [], "val": []})  # ← restaurar historial
        # Restaurar también el best_state desde el _best.pth si existe
        best_ckpt = CHECKPOINTS_DIR / f"{model_name}_best.pth"
        if best_ckpt.exists():
            best_data  = torch.load(best_ckpt, map_location=device, weights_only=True)
            best_state = best_data["model_state"]
            best_epoch = best_data["epoch"]
        print(f"  Reanudando desde epoch {start_epoch} | best_miou: {best_miou:.4f} | "
              f"epochs en historial: {len(history['train'])}")
    else:
        print(f"  Entrenamiento desde cero")

    print(f"\n{'='*60}")
    print(f"  {model_name} | device={device} | AMP={use_amp}")
    print(f"  epochs={num_epochs} | patience={patience} | aux_weight={aux_weight}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                             device, scaler=scaler, aux_weight=aux_weight)
        va = evaluate(model, val_loader, criterion, device, scaler=scaler)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["mIoU"])
            else:
                scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)

        print(f"Ep {epoch:3d}/{num_epochs} | "
              f"loss={tr['loss']:.4f} mIoU={tr['mIoU']:.4f} | "
              f"val_loss={va['loss']:.4f} val_mIoU={va['mIoU']:.4f} | "
              f"rock={va['IoU_per_class']['big_rock']:.4f}")

        # Checkpoint del mejor modelo
        if va["mIoU"] > best_miou:
            best_miou  = va["mIoU"]
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"epoch": epoch, "model_state": best_state, "val_metrics": va},
                CHECKPOINTS_DIR / f"{model_name}_best.pth"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping en epoch {epoch} "
                      f"(sin mejora por {patience} epochs)")
                break

        # Checkpoint periódico cada 3 epochs — sobreescribe siempre el mismo archivo
        if epoch % 3 == 0:
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "best_miou":       best_miou,
                    "no_improve":      no_improve,
                    "history":         history,   # ← guardar historial completo
                },
                periodic_ckpt,
            )
            print(f"Checkpoint periódico guardado (epoch {epoch})")

    train_time = time.time() - t_start
    print(f"\n  Mejor val mIoU={best_miou:.4f} (epoch {best_epoch}) | "
          f"tiempo={train_time:.0f}s")

    # Restaurar mejor estado
    model.load_state_dict(best_state)

    # Limpiar checkpoint periódico al terminar exitosamente
    if periodic_ckpt.exists():
        periodic_ckpt.unlink()
        print(f" Checkpoint periódico eliminado")

    return history, best_miou, best_epoch, train_time


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SEED  —  3 runs, reporta media ± std e IC95%
# ─────────────────────────────────────────────────────────────────────────────
def _serialize_history(history: dict) -> dict:
    """
    Convierte el dict history a tipos nativos Python para serialización JSON.
    Maneja float, numpy.floating, dict de floats (IoU_per_class) y otros valores.
    """
    def _to_native(v):
        if isinstance(v, dict):
            return {ck: float(cv) for ck, cv in v.items()}
        if isinstance(v, (float, int)):
            return v
        try:
            return float(v)   # captura numpy.float32/64, torch scalars, etc.
        except (TypeError, ValueError):
            return v          # fallback: dejar tal cual (str, None, etc.)

    return {
        split: [
            {k: _to_native(val) for k, val in epoch_dict.items()}
            for epoch_dict in epochs
        ]
        for split, epochs in history.items()
    }


def run_multi_seed(model_fn, df_train, df_val, df_gold,
                   criterion_fn, optimizer_fn, scheduler_fn,
                   model_name: str, device,
                   seeds=None,
                   num_epochs:  int   = 80,
                   patience:    int   = 10,
                   batch_size:  int   = BATCH_SIZE,
                   num_workers: int   = 4,
                   aux_weight: float  = 0.0,
                   use_amp:    bool   = True,
                   fast_subset: bool  = False,
                   n_train_fast: int  = 200,
                   n_val_fast:   int  = 50) -> dict:

    if seeds is None:
        seeds = SEEDS

    mean, std = load_norm_stats()

    # ── Cargar progreso previo si existe ─────────────────────────────────────
    progress_path = CHECKPOINTS_DIR / f"{model_name}_progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        all_test       = progress["completed_seeds"]
        completed_seeds = {r["seed"] for r in all_test}
        print(f"Progreso encontrado — seeds completados: {completed_seeds}")
    else:
        all_test        = []
        completed_seeds = set()

    for seed in seeds:
        if seed in completed_seeds:
            print(f"\n  Seed {seed} ya completado — saltando")
            continue

        set_seed(seed)
        print(f"\n{'─'*55}")
        print(f"  Seed {seed} | {model_name}")

        if fast_subset:
            tr, va = make_fast_subset(df_train, df_val,
                                      n_train=n_train_fast,
                                      n_val=n_val_fast,
                                      seed=seed)
        else:
            tr, va = df_train, df_val

        train_loader, val_loader, gold_loader = build_dataloaders(
            tr, va, df_gold,
            mean=mean, std=std,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model     = model_fn().to(device)
        criterion = criterion_fn()
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        history, best_miou, best_epoch, train_time = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion, device,
            model_name=f"{model_name}_seed{seed}",
            num_epochs=num_epochs,
            patience=patience,
            aux_weight=aux_weight,
            use_amp=use_amp,
        )

        # Evaluar sobre gold test set
        test_res = evaluate(model, gold_loader, criterion, device)
        test_res.update({
            "seed":       seed,
            "best_epoch": best_epoch,
            "train_time": train_time,
            "history":    history,
        })
        all_test.append(test_res)
        print(f"  Gold test mIoU={test_res['mIoU']:.4f}")

        # ── Guardar progreso tras cada seed completado ────────────────────
        # Serializar history a tipos nativos Python para JSON
        progress_data = {
            "completed_seeds": [
                {**{k: v for k, v in r.items() if k != "history"},
                 "history": _serialize_history(r["history"])}
                for r in all_test
            ]
        }
        with open(progress_path, "w") as f:
            json.dump(progress_data, f)
        print(f" Progreso guardado ({len(all_test)}/{len(seeds)} seeds)")

        # ── Borrar checkpoint periódico del seed recién completado ────────
        periodic_ckpt = CHECKPOINTS_DIR / f"{model_name}_seed{seed}_periodic.pth"
        if periodic_ckpt.exists():
            periodic_ckpt.unlink()
            print(f" Checkpoint periódico eliminado")

    # ── Resumen estadístico ───────────────────────────────────────────────────
    miou_vals = [r["mIoU"] for r in all_test]
    summary   = {
        "model":           model_name,
        "mIoU_mean":       float(np.mean(miou_vals)),
        "mIoU_std":        float(np.std(miou_vals)),
        "mIoU_ci95":       float(1.96 * np.std(miou_vals) / np.sqrt(len(miou_vals))),
        "pixel_acc_mean":  float(np.mean([r["pixel_accuracy"] for r in all_test])),
        "train_time_mean": float(np.mean([r["train_time"]     for r in all_test])),
        "best_epoch_mean": float(np.mean([r["best_epoch"]     for r in all_test])),
        "per_seed":        all_test,
    }
    for c in CLASS_NAMES.values():
        vals = [r["IoU_per_class"][c] for r in all_test]
        summary[f"iou_{c}_mean"] = float(np.mean(vals))
        summary[f"iou_{c}_std"]  = float(np.std(vals))

    print(f"\n{'='*60}")
    print(f"  {model_name}: mIoU={summary['mIoU_mean']:.4f} ± {summary['mIoU_std']:.4f}"
          f"  IC95=±{summary['mIoU_ci95']:.4f}")
    for c in CLASS_NAMES.values():
        print(f"  IoU({c:10s})={summary[f'iou_{c}_mean']:.4f} ± {summary[f'iou_{c}_std']:.4f}")
    print(f"{'='*60}")

    # ── Limpiar progreso al completar todos los seeds ─────────────────────────
    if progress_path.exists():
        progress_path.unlink()
        print(f" Archivo de progreso eliminado (todos los seeds completados)")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO CSV  —  agrega fila al benchmark_results.csv compartido
# ─────────────────────────────────────────────────────────────────────────────
def append_benchmark_results(summary: dict, params_M: float):
    """
    Agrega una fila al CSV de benchmark con los resultados del modelo.

    Parameters
    ----------
    summary  : dict devuelto por run_multi_seed
    params_M : número de parámetros en millones (count_parameters(model))
    """
    fieldnames = [
        "model", "mIoU_mean", "mIoU_std", "mIoU_ci95",
        "pixel_acc_mean",
        "iou_soil_mean",    "iou_soil_std",
        "iou_bedrock_mean", "iou_bedrock_std",
        "iou_sand_mean",    "iou_sand_std",
        "iou_big_rock_mean","iou_big_rock_std",
        "params_M", "train_time_mean_s", "best_epoch_mean",
    ]
    row = {
        "model":              summary["model"],
        "mIoU_mean":          round(summary["mIoU_mean"],       4),
        "mIoU_std":           round(summary["mIoU_std"],        4),
        "mIoU_ci95":          round(summary["mIoU_ci95"],       4),
        "pixel_acc_mean":     round(summary["pixel_acc_mean"],  4),
        "iou_soil_mean":      round(summary["iou_soil_mean"],   4),
        "iou_soil_std":       round(summary["iou_soil_std"],    4),
        "iou_bedrock_mean":   round(summary["iou_bedrock_mean"],4),
        "iou_bedrock_std":    round(summary["iou_bedrock_std"], 4),
        "iou_sand_mean":      round(summary["iou_sand_mean"],   4),
        "iou_sand_std":       round(summary["iou_sand_std"],    4),
        "iou_big_rock_mean":  round(summary["iou_big_rock_mean"],4),
        "iou_big_rock_std":   round(summary["iou_big_rock_std"], 4),
        "params_M":           round(params_M,                   2),
        "train_time_mean_s":  round(summary["train_time_mean"], 0),
        "best_epoch_mean":    round(summary["best_epoch_mean"], 1),
    }
    write_header = not BENCHMARK_CSV.exists()
    with open(BENCHMARK_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Resultados guardados en {BENCHMARK_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# CURVAS DE ENTRENAMIENTO  —  solo plt.show(), sin savefig
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, model_name: str = ""):
    """
    Grafica loss y mIoU de train y val por epoch.
    Solo plt.show() — sin savefig (política del proyecto).
    """
    train_loss = [e["loss"]  for e in history["train"]]
    val_loss   = [e["loss"]  for e in history["val"  ]]
    train_miou = [e["mIoU"]  for e in history["train"]]
    val_miou   = [e["mIoU"]  for e in history["val"  ]]
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Curvas de entrenamiento — {model_name}", fontweight="bold")

    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss,   label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_miou, label="Train mIoU")
    axes[1].plot(epochs, val_miou,   label="Val mIoU")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title("mIoU"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN DE PREDICCIONES  —  solo plt.show(), sin savefig
# ─────────────────────────────────────────────────────────────────────────────
def visualize_predictions(model, df_gold: pd.DataFrame, device,
                           mean, std, n: int = 5):
    """
    Muestra n ejemplos del gold test set: imagen original | GT | predicción.
    Solo plt.show() — sin savefig (política del proyecto).
    """
    def mask_to_rgb(m):
        rgb = np.zeros((*m.shape, 3))
        for c, col in CLASS_RGB.items():
            rgb[m == c] = col
        return rgb

    sample_df = df_gold.sample(n, random_state=42).reset_index(drop=True)
    ds        = MarsTerrainDataset(sample_df, JointTransformVal(mean, std))

    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Predicciones vs Ground Truth", fontsize=14, fontweight="bold")

    mean_arr = np.array(mean)
    std_arr  = np.array(std)

    with torch.no_grad():
        for i in range(n):
            img_t, mask_t = ds[i]
            out    = model(img_t.unsqueeze(0).to(device))
            logits = out["out"] if isinstance(out, dict) else out
            pred   = logits.argmax(1).squeeze().cpu().numpy()
            gt     = mask_t.numpy()

            # Desnormalizar imagen para visualizar
            img_np = np.clip(
                img_t.permute(1, 2, 0).numpy() * std_arr + mean_arr, 0, 1
            )

            axes[i][0].imshow(img_np[:, :, 0], cmap="gray")
            axes[i][0].set_title("Imagen"); axes[i][0].axis("off")
            axes[i][1].imshow(mask_to_rgb(gt))
            axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
            axes[i][2].imshow(mask_to_rgb(pred))
            axes[i][2].set_title("Predicción"); axes[i][2].axis("off")

    patches = [mpatches.Patch(color=CLASS_RGB[c], label=CLASS_NAMES[c])
               for c in range(4)]
    patches.append(mpatches.Patch(color=CLASS_RGB[255], label="ignore"))
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.show()

def plot_best_seed_curves(summary: dict):
    """
    Grafica curvas de entrenamiento del seed con mejor mIoU en gold test.
    Solo plt.show() — sin savefig.
    """
    best = max(summary["per_seed"], key=lambda r: r["mIoU"])
    plot_training_curves(best["history"], 
                         model_name=f"{summary['model']} (seed {best['seed']})")


def print_summary_table(summary: dict):
    """
    Imprime tabla resumen agregada de los 3 seeds.
    """
    print(f"\n{'='*60}")
    print(f"  RESUMEN — {summary['model']}")
    print(f"{'='*60}")
    print(f"  {'Métrica':<20} {'Media':>8}  {'Std':>8}  {'IC95':>8}")
    print(f"  {'-'*48}")
    print(f"  {'mIoU':<20} {summary['mIoU_mean']:>8.4f}  "
          f"{summary['mIoU_std']:>8.4f}  ±{summary['mIoU_ci95']:>7.4f}")
    print(f"  {'Pixel Accuracy':<20} {summary['pixel_acc_mean']:>8.4f}")
    print(f"  {'-'*48}")
    for c in CLASS_NAMES.values():
        print(f"  {f'IoU {c}':<20} {summary[f'iou_{c}_mean']:>8.4f}  "
              f"{summary[f'iou_{c}_std']:>8.4f}")
    print(f"  {'-'*48}")
    print(f"  {'Params (M)':<20} —")
    print(f"  {'Tiempo medio (s)':<20} {summary['train_time_mean']:>8.0f}")
    print(f"  {'Epoch mejor (mean)':<20} {summary['best_epoch_mean']:>8.1f}")
    print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model) -> float:
    """Devuelve el número de parámetros entrenables en millones."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
