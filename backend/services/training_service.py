"""
TrainingManager — Serviço centralizado de treinamento de modelos.

Executa o treinamento em thread separada, reportando progresso
via callback (consumido pelo WebSocket).
"""

import os
import random
import sys
import time
import threading
import json
from datetime import datetime
from typing import Callable, Optional

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B7_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
)

# ──────────────────────── Configuração de modelo ────────────────────────

TRAINING_MODEL_CONFIGS = {
    "EfficientNetB0": {
        "builder": torchvision.models.efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "input_size": 224,
        "classifier_type": "sequential",  # model.classifier[1]
        "default_batch": 16,
        "cpu_batch": 4,
    },
    "EfficientNetB2": {
        "builder": torchvision.models.efficientnet_b2,
        "weights": EfficientNet_B2_Weights.IMAGENET1K_V1,
        "input_size": 260,
        "classifier_type": "sequential",
        "default_batch": 16,
        "cpu_batch": 4,
    },
    "EfficientNetB3": {
        "builder": torchvision.models.efficientnet_b3,
        "weights": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "input_size": 300,
        "classifier_type": "sequential",
        "default_batch": 16,
        "cpu_batch": 4,
    },
    "EfficientNetB7": {
        "builder": torchvision.models.efficientnet_b7,
        "weights": EfficientNet_B7_Weights.IMAGENET1K_V1,
        "input_size": 600,
        "classifier_type": "sequential",  # model.classifier[1]
        "default_batch": 8,
        "cpu_batch": 1,
    },
    "ResNet50": {
        "builder": torchvision.models.resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "input_size": 224,
        "classifier_type": "fc",  # model.fc
        "default_batch": 16,
        "cpu_batch": 4,
    },
    "MobileNetV3": {
        "builder": torchvision.models.mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "input_size": 224,
        "classifier_type": "sequential",  # model.classifier[-1]
        "default_batch": 32,
        "cpu_batch": 8,
    },
}

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_SAVE_DIR = os.path.join(WORKSPACE_ROOT, "models")

# ──────────────────────── Runtime helpers ────────────────────────


def _configure_runtime():
    """Configura device e paralelismo CPU/GPU."""
    cpu_threads = os.cpu_count() or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_windows = sys.platform.startswith("win")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except RuntimeError:
            pass  # já configurado

    # No Windows, workers extras fazem spawn de novos processos que
    # reimportam torch/sklearn/pandas e pressionam muito RAM/pagefile.
    if is_windows:
        num_workers = 0
    else:
        num_workers = (
            min(4, max(1, cpu_threads // 4))
            if device.type == "cpu"
            else min(8, cpu_threads)
        )
    pin_memory = device.type == "cuda"
    persistent_workers = device.type == "cuda" and num_workers > 0
    prefetch_factor = 1 if num_workers > 0 else 2

    return device, num_workers, pin_memory, persistent_workers, prefetch_factor


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(model_name: str, num_classes: int, device: torch.device):
    """Constrói o modelo com pesos pré-treinados e ajusta a camada final."""
    cfg = TRAINING_MODEL_CONFIGS[model_name]
    model = cfg["builder"](weights=cfg["weights"])

    if cfg["classifier_type"] == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # EfficientNet / MobileNet — última Linear no Sequential
        for idx in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[idx], nn.Linear):
                in_features = model.classifier[idx].in_features
                model.classifier[idx] = nn.Linear(in_features, num_classes)
                break

    return model.to(device)


def _freeze_backbone(model: torch.nn.Module, classifier_type: str, freeze: bool):
    for param in model.parameters():
        param.requires_grad = not freeze

    if classifier_type == "fc":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True


def _build_optimizer(config: dict, model: torch.nn.Module):
    optimizer_name = config.get("optimizer_name", "AdamW")
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("Nenhum parâmetro treinável encontrado para o otimizador.")

    if optimizer_name != "AdamW":
        raise ValueError(f"Otimizador não suportado: '{optimizer_name}'.")

    return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(config: dict, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 2),
        min_lr=config.get("scheduler_min_lr", 1e-6),
    )


class TransformSubset(torch.utils.data.Dataset):
    """Garante que Apply/Transforms ocorram *após* carregar do ImageFolder base."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def _split_class_indices(
    indices: list[int],
    train_split: float,
    val_split: float,
) -> tuple[list[int], list[int], list[int]]:
    n_items = len(indices)
    if n_items < 3:
        raise ValueError(
            "Split estratificado exige pelo menos 3 imagens por classe. "
            f"Classe com {n_items} imagem(ns) encontrada."
        )

    test_split = 1.0 - train_split - val_split
    val_count = max(1, int(round(val_split * n_items)))
    test_count = max(1, int(round(test_split * n_items)))
    train_count = n_items - val_count - test_count

    while train_count < 1 and (val_count > 1 or test_count > 1):
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        train_count = n_items - val_count - test_count

    if train_count < 1:
        raise ValueError(
            "Nao foi possivel criar split estratificado com pelo menos uma "
            f"amostra de treino para classe com {n_items} imagem(ns)."
        )

    train_end = train_count
    val_end = train_end + val_count
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _stratified_split(
    dataset: ImageFolder,
    num_classes: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0].tolist()
        rng.shuffle(class_indices)
        cls_train, cls_val, cls_test = _split_class_indices(
            class_indices,
            train_split,
            val_split,
        )
        train_indices.extend(cls_train)
        val_indices.extend(cls_val)
        test_indices.extend(cls_test)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def _macro_f1_score(labels: list[int], preds: list[int], num_classes: int) -> float:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    f1_scores = []

    for class_idx in range(num_classes):
        true_positive = np.sum((preds_np == class_idx) & (labels_np == class_idx))
        false_positive = np.sum((preds_np == class_idx) & (labels_np != class_idx))
        false_negative = np.sum((preds_np != class_idx) & (labels_np == class_idx))

        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _is_checkpoint_improved(metric_name: str, current: float, best: float) -> bool:
    if metric_name == "val_loss":
        return current < best
    return current > best


def _safe_name(value: str) -> str:
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    )
    return "_".join(part for part in safe.split("_") if part)


def _build_train_transforms(input_size: int, augmentation_profile: str = "standard"):
    """Transforms de treinamento configuraveis por perfil."""
    profiles = {
        # Mantem o comportamento historico da pipeline.
        "standard": {
            "scale": (0.8, 1.0),
            "rotation": 15,
            "color_jitter": dict(brightness=0.2, contrast=0.2, saturation=0.2),
        },
        # Menos agressivo em cor e crop para classes onde cor e formato fino
        # carregam sinal semantico, como ESVERDEADO, PURPURAS e CHOCHOS.
        "conservative_color": {
            "scale": (0.9, 1.0),
            "rotation": 10,
            "color_jitter": dict(brightness=0.08, contrast=0.08, saturation=0.08),
        },
        # Remove jitter de cor para testar se variacao cromatica esta apagando
        # fronteiras entre classes dependentes de cor.
        "no_color_jitter": {
            "scale": (0.9, 1.0),
            "rotation": 10,
            "color_jitter": None,
        },
    }
    profile = profiles.get(augmentation_profile)
    if profile is None:
        raise ValueError(
            "augmentation_profile deve ser 'standard', 'conservative_color' "
            f"ou 'no_color_jitter'. Recebido: {augmentation_profile}"
        )

    transforms = [
        T.RandomResizedCrop(input_size, scale=profile["scale"]),
        T.RandomHorizontalFlip(),
        T.RandomRotation(profile["rotation"]),
    ]
    if profile["color_jitter"]:
        transforms.append(T.ColorJitter(**profile["color_jitter"]))
    transforms.extend([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return T.Compose(transforms)


def _build_val_transforms(input_size: int):
    """Transforms rígidos sem distorções para Teste e Validação."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _build_class_weights(
    class_counts: np.ndarray,
    strategy: str = "sqrt_inverse",
    effective_beta: float = 0.999,
) -> np.ndarray:
    """Calcula pesos de classe normalizados para media aproximada 1."""
    counts = class_counts.astype(np.float64)

    if strategy == "none":
        weights = np.ones_like(counts, dtype=np.float64)
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts + 1e-8)
    elif strategy == "inverse":
        weights = 1.0 / (counts + 1e-8)
    elif strategy == "effective_number":
        beta = float(effective_beta)
        if not 0.0 < beta < 1.0:
            raise ValueError("effective_number_beta deve estar entre 0 e 1.")
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    else:
        raise ValueError(
            "class_weight_strategy deve ser 'sqrt_inverse', 'inverse', "
            f"'effective_number' ou 'none'. Recebido: {strategy}"
        )

    weights = weights / np.sum(weights) * len(weights)
    return weights


class FocalLoss(nn.Module):
    """Focal Loss multi-classe com pesos e label smoothing opcionais."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def _build_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 1.5,
) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )
    if loss_name == "focal":
        return FocalLoss(
            weight=class_weights,
            gamma=float(focal_gamma),
            label_smoothing=float(label_smoothing),
        )
    raise ValueError(
        f"loss_name deve ser 'cross_entropy' ou 'focal'. Recebido: {loss_name}"
    )


# ──────────────────────── Dataset scanning ────────────────────────


def scan_datasets(base_dir: Optional[str] = None):
    """Lista diretórios de dataset disponíveis com labels e contagens."""
    if base_dir is None:
        base_dir = os.path.join(WORKSPACE_ROOT, "data")

    datasets = []
    if not os.path.isdir(base_dir):
        return datasets

    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # Verifica se contém subpastas (labels)
        labels = {}
        for sub in sorted(os.listdir(full_path)):
            sub_path = os.path.join(full_path, sub)
            if os.path.isdir(sub_path):
                # Conta imagens na subpasta (classes)
                count = sum(
                    1
                    for f in os.listdir(sub_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
                )
                if count > 0:
                    labels[sub] = count

        # Quantas imagens estão soltas diretamente na raiz desta pasta? (Erro comum do usuário)
        root_images_count = sum(
            1 for f in os.listdir(full_path)
            if not os.path.isdir(os.path.join(full_path, f)) and 
            f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
        )

        datasets.append({
            "name": entry,
            "path": full_path,
            "labels": labels,
            "total_images": sum(labels.values()) + root_images_count,
            "root_images": root_images_count,
        })

    return datasets


# ──────────────────────── TrainingManager ────────────────────────


class TrainingManager:
    """Gerencia o ciclo de vida de um treinamento (1 por vez)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._stop_early_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.is_training = False
        self.is_paused = False
        self.last_result: Optional[dict] = None

    # ── public API ──

    def start(self, config: dict, progress_callback: Callable):
        """Inicia treinamento em thread separada.

        Args:
            config: dict com model_name, data_path, batch_size, num_epochs,
                    learning_rate, patience, train_split, val_split
            progress_callback: chamado com dict de progresso
        """
        with self._lock:
            if self.is_training:
                raise RuntimeError("Já existe um treinamento em andamento.")

            self._cancel_event.clear()
            self._stop_early_event.clear()
            self.is_paused = False
            self.is_training = True
            self.last_result = None

        self._thread = threading.Thread(
            target=self._run,
            args=(config, progress_callback),
            daemon=True,
        )
        self._thread.start()

    def run_blocking(self, config: dict, progress_callback: Callable):
        """Executa um treinamento no processo atual, sem thread auxiliar.

        Útil para scripts CLI/pipelines onde os jobs devem rodar em sequência.
        """
        with self._lock:
            if self.is_training:
                raise RuntimeError("Já existe um treinamento em andamento.")

            self._cancel_event.clear()
            self._stop_early_event.clear()
            self.is_paused = False
            self.is_training = True
            self.last_result = None

        try:
            self._train_loop(config, progress_callback)
        finally:
            with self._lock:
                self.is_training = False

    def cancel(self):
        """Sinaliza cancelamento do treinamento (Descarta)."""
        self.is_paused = False
        self._cancel_event.set()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop_early(self):
        """Finaliza e salva o estado atual antes das épocas terminarem."""
        self.is_paused = False
        self._stop_early_event.set()

    @property
    def status(self) -> str:
        if self.is_training:
            if self.is_paused:
                return "paused"
            if self._stop_early_event.is_set():
                return "stopping"
            return "training"
        if self.last_result is not None:
            return "completed"
        return "idle"

    # ── internal ──

    def _run(self, config: dict, callback: Callable):
        try:
            self._train_loop(config, callback)
        except Exception as exc:
            callback({"type": "training_error", "message": str(exc)})
        finally:
            with self._lock:
                self.is_training = False

    def _train_loop(self, config: dict, callback: Callable):
        model_name = config["model_name"]
        experiment_name = config.get("experiment_name", "default")
        data_path = config["data_path"]
        batch_size = config.get("batch_size", 16)
        num_epochs = config.get("num_epochs", 20)
        patience = config.get("patience", 5)
        early_stopping = bool(config.get("early_stopping", True))
        split_strategy = config.get("split_strategy", "random")
        checkpoint_metric = config.get("checkpoint_metric", "val_loss")
        sampler_strategy = config.get("sampler_strategy", "shuffle")
        loss_name = config.get("loss_name", "cross_entropy")
        class_weight_strategy = config.get("class_weight_strategy", "sqrt_inverse")
        label_smoothing = float(config.get("label_smoothing", 0.0))
        focal_gamma = float(config.get("focal_gamma", 1.5))
        effective_number_beta = float(config.get("effective_number_beta", 0.999))
        augmentation_profile = config.get("augmentation_profile", "standard")
        train_split = config.get("train_split", 0.8)
        val_split = config.get("val_split", 0.1)
        seed = int(config.get("seed", 42))
        accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))
        freeze_backbone_epochs = max(0, int(config.get("freeze_backbone_epochs", 0)))
        fine_tune_learning_rate = config.get(
            "fine_tune_learning_rate",
            config.get("learning_rate", 1e-4),
        )

        if model_name not in TRAINING_MODEL_CONFIGS:
            raise ValueError(f"Modelo '{model_name}' não suportado.")
        if split_strategy not in {"random", "stratified"}:
            raise ValueError(
                "split_strategy deve ser 'random' ou 'stratified'. "
                f"Recebido: {split_strategy}"
            )
        if checkpoint_metric not in {"val_loss", "val_accuracy", "val_macro_f1"}:
            raise ValueError(
                "checkpoint_metric deve ser 'val_loss', 'val_accuracy' ou "
                f"'val_macro_f1'. Recebido: {checkpoint_metric}"
            )
        if sampler_strategy not in {"shuffle", "weighted"}:
            raise ValueError(
                "sampler_strategy deve ser 'shuffle' ou 'weighted'. "
                f"Recebido: {sampler_strategy}"
            )
        if loss_name not in {"cross_entropy", "focal"}:
            raise ValueError(
                "loss_name deve ser 'cross_entropy' ou 'focal'. "
                f"Recebido: {loss_name}"
            )
        if class_weight_strategy not in {
            "sqrt_inverse",
            "inverse",
            "effective_number",
            "none",
        }:
            raise ValueError(
                "class_weight_strategy deve ser 'sqrt_inverse', 'inverse', "
                "'effective_number' ou 'none'. "
                f"Recebido: {class_weight_strategy}"
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing deve ser >= 0 e < 1.")
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma deve ser >= 0.")
        if augmentation_profile not in {
            "standard",
            "conservative_color",
            "no_color_jitter",
        }:
            raise ValueError(
                "augmentation_profile deve ser 'standard', "
                "'conservative_color' ou 'no_color_jitter'. "
                f"Recebido: {augmentation_profile}"
            )

        cfg = TRAINING_MODEL_CONFIGS[model_name]
        _set_seed(seed)

        # ── Device & runtime ──
        device, num_workers, pin_memory, persistent_workers, prefetch_factor = (
            _configure_runtime()
        )

        # Ajustar batch_size para CPU se necessário
        if device.type == "cpu":
            max_cpu_batch = cfg["cpu_batch"]
            if batch_size > max_cpu_batch:
                batch_size = max_cpu_batch

        callback({
            "type": "status",
            "message": f"Configurando {model_name} no {device.type.upper()}...",
        })

        # ── Dataset ──
        # Carrega as imagens puras (sem transformações globais prejudiciais pra Val)
        dataset = ImageFolder(root=data_path, transform=None)
        num_classes = len(dataset.classes)
        class_names = dataset.classes

        total = len(dataset)
        train_size = int(train_split * total)
        val_size = int(val_split * total)
        test_size = total - train_size - val_size

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(
                f"Dataset muito pequeno ({total} imagens) para os splits configurados."
            )

        if split_strategy == "stratified":
            train_ds_raw, val_ds_raw, test_ds_raw = _stratified_split(
                dataset,
                num_classes,
                train_split,
                val_split,
                seed,
            )
            train_size = len(train_ds_raw)
            val_size = len(val_ds_raw)
            test_size = len(test_ds_raw)
        else:
            split_generator = torch.Generator().manual_seed(seed)
            train_ds_raw, val_ds_raw, test_ds_raw = random_split(
                dataset, [train_size, val_size, test_size], generator=split_generator
            )

        train_transforms = _build_train_transforms(
            cfg["input_size"],
            augmentation_profile,
        )
        val_transforms = _build_val_transforms(cfg["input_size"])

        train_ds = TransformSubset(train_ds_raw, transform=train_transforms)
        val_ds = TransformSubset(val_ds_raw, transform=val_transforms)
        test_ds = TransformSubset(test_ds_raw, transform=val_transforms)

        # ── Class Weights ──
        callback({"type": "status", "message": "Calculando os pesos das classes..."})
        train_indices = train_ds_raw.indices
        targets = [dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets, minlength=num_classes)
        
        weights = _build_class_weights(
            class_counts,
            class_weight_strategy,
            effective_number_beta,
        )
        class_weights = (
            None
            if class_weight_strategy == "none"
            else torch.FloatTensor(weights).to(device)
        )
        sample_weights = torch.DoubleTensor([weights[target] for target in targets])

        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

        if sampler_strategy == "weighted":
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
            )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
        )

        callback({
            "type": "status",
            "message": (
                f"Dataset carregado: {total} imagens, {num_classes} classes. "
                f"Train={train_size}, Val={val_size}, Test={test_size}"
            ),
        })

        # ── Model ──
        model = _build_model(model_name, num_classes, device)
        if freeze_backbone_epochs > 0:
            _freeze_backbone(model, cfg["classifier_type"], freeze=True)
        criterion = _build_criterion(
            loss_name,
            class_weights,
            label_smoothing,
            focal_gamma,
        )
        optimizer = _build_optimizer(config, model)
        scheduler = _build_scheduler(config, optimizer)

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # ── Training loop ──
        best_val_loss = float("inf")
        best_checkpoint_score = (
            float("inf") if checkpoint_metric == "val_loss" else float("-inf")
        )
        best_epoch = 0
        epochs_no_improve = 0
        total_batches = len(train_loader)
        start_time = time.time()
        train_images_seen = 0
        epoch_history = []
        current_phase = "head" if freeze_backbone_epochs > 0 else "full"
        phase_switched = False

        # Caminho para salvar pesos com timestamp
        os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = _safe_name(
            config.get("dataset_name") or os.path.basename(os.path.normpath(data_path))
        )
        experiment_slug = _safe_name(experiment_name)
        model_file_prefix = f"soybean_model_{model_name.lower()}_{dataset_slug}"
        if experiment_slug and experiment_slug != "default":
            model_file_prefix = f"{model_file_prefix}_{experiment_slug}"
        save_path = os.path.join(
            MODELS_SAVE_DIR, f"{model_file_prefix}_{timestamp_str}.pth"
        )

        has_saved_checkpoint = False

        for epoch in range(num_epochs):

            while self.is_paused:
                if self._cancel_event.is_set() or self._stop_early_event.is_set():
                    break
                time.sleep(0.5)

            if self._cancel_event.is_set():
                callback({
                    "type": "training_cancelled",
                    "epochs_completed": epoch,
                    "has_checkpoint": has_saved_checkpoint,
                    "checkpoint_path": save_path if has_saved_checkpoint else None,
                })
                return
                
            if self._stop_early_event.is_set():
                callback({"type": "status", "message": f"Treino finalizado antecipadamente. Salvando da época {epoch}..."})
                break

            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs and not phase_switched:
                callback({
                    "type": "status",
                    "message": "Mudando para fase 2: destravando backbone para fine-tuning completo.",
                })
                _freeze_backbone(model, cfg["classifier_type"], freeze=False)
                fine_tune_config = dict(config)
                fine_tune_config["learning_rate"] = fine_tune_learning_rate
                optimizer = _build_optimizer(fine_tune_config, model)
                scheduler = _build_scheduler(config, optimizer)
                current_phase = "full"
                phase_switched = True
                epochs_no_improve = 0

            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (data, targets) in enumerate(train_loader):
                while self.is_paused:
                    if self._cancel_event.is_set() or self._stop_early_event.is_set():
                        break
                    time.sleep(0.5)

                if self._cancel_event.is_set():
                    callback({
                        "type": "training_cancelled",
                        "epochs_completed": epoch,
                        "has_checkpoint": has_saved_checkpoint,
                        "checkpoint_path": save_path if has_saved_checkpoint else None,
                    })
                    return
                
                if self._stop_early_event.is_set():
                    break

                data = data.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss_for_backward = loss / accumulation_steps

                scaler.scale(loss_for_backward).backward()

                should_step = (
                    (batch_idx + 1) % accumulation_steps == 0
                    or batch_idx == total_batches - 1
                )
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += loss.item()
                train_images_seen += int(data.size(0))

                # Progresso por batch (a cada 5 batches para não sobrecarregar)
                if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                    callback({
                        "type": "batch_progress",
                        "epoch": epoch + 1,
                        "total_epochs": num_epochs,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "loss": loss.item(),
                    })

            epoch_train_loss = running_loss / total_batches
            
            if self._stop_early_event.is_set():
                callback({"type": "status", "message": f"Batches interrompidos. Finalizando e avaliando..."})
                break

            # ── Validação ──
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_accuracy = (
                float(np.mean(np.asarray(val_preds) == np.asarray(val_labels)))
                if val_labels
                else 0.0
            )
            epoch_val_macro_f1 = _macro_f1_score(val_labels, val_preds, num_classes)
            checkpoint_score = {
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_accuracy,
                "val_macro_f1": epoch_val_macro_f1,
            }[checkpoint_metric]
            elapsed = time.time() - start_time
            scheduler.step(epoch_val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            epoch_history.append({
                "epoch": epoch + 1,
                "phase": current_phase,
                "train_loss": round(epoch_train_loss, 6),
                "val_loss": round(epoch_val_loss, 6),
                "val_accuracy": round(epoch_val_accuracy * 100, 4),
                "val_macro_f1": round(epoch_val_macro_f1 * 100, 4),
                "checkpoint_score": round(checkpoint_score, 6),
                "learning_rate": current_lr,
                "elapsed_seconds": round(elapsed, 1),
            })

            callback({
                "type": "epoch_complete",
                "epoch": epoch + 1,
                "total_epochs": num_epochs,
                "train_loss": round(epoch_train_loss, 6),
                "val_loss": round(epoch_val_loss, 6),
                "val_accuracy": round(epoch_val_accuracy * 100, 4),
                "val_macro_f1": round(epoch_val_macro_f1 * 100, 4),
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_score": round(checkpoint_score, 6),
                "elapsed_seconds": round(elapsed, 1),
                "learning_rate": current_lr,
                "phase": current_phase,
            })

            # ── Checkpoint / early stopping ──
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            if _is_checkpoint_improved(
                checkpoint_metric, checkpoint_score, best_checkpoint_score
            ):
                best_checkpoint_score = checkpoint_score
                best_epoch = epoch + 1
                epochs_no_improve = 0
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_score": checkpoint_score,
                    "epoch": best_epoch,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_accuracy,
                    "val_macro_f1": epoch_val_macro_f1,
                }
                torch.save(checkpoint, save_path)
                has_saved_checkpoint = True
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    callback({
                        "type": "status",
                        "message": f"Early stopping na epoch {epoch + 1}",
                    })
                    break

        # ── Avaliação no test set ──
        callback({"type": "status", "message": "Avaliando modelo no conjunto de teste..."})

        # Import tardio: evita que subprocessos do DataLoader no Windows
        # carreguem sklearn/pandas desnecessariamente.
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        eval_started_at = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=pin_memory)
                outputs = model(images)
                
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        eval_time = time.time() - eval_started_at

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True,
            digits=4,
        )

        # Accuracy
        accuracy = report.get("accuracy", 0.0)
        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)

        # Per-class metrics
        per_class = []
        for cname in class_names:
            if cname in report:
                per_class.append({
                    "class": cname,
                    "precision": round(report[cname]["precision"] * 100, 2),
                    "recall": round(report[cname]["recall"] * 100, 2),
                    "f1": round(report[cname]["f1-score"] * 100, 2),
                    "support": int(report[cname]["support"]),
                })

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        # ROC Curves
        roc_curves = []
        common_fpr = np.linspace(0, 1, 100)
        
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        
        for i, cname in enumerate(class_names):
            y_true_binary = (all_labels_np == i).astype(int)
            y_score = all_probs_np[:, i]
            
            # Avoid errors if a class has no positive samples in test slice
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Interpolate to have a uniform FPR axis across all classes
                interp_tpr = np.interp(common_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                
                roc_curves.append({
                    "class": cname,
                    "tpr": np.round(interp_tpr, 4).tolist(),
                    "auc": float(round(roc_auc, 4))
                })

        total_time = time.time() - start_time
        parameter_count = sum(p.numel() for p in model.parameters())
        trainable_parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model_size_mb = (
            os.path.getsize(save_path) / (1024 * 1024)
            if os.path.exists(save_path)
            else 0.0
        )

        result = {
            "type": "training_complete",
            "experiment_name": experiment_name,
            "dataset_name": config.get("dataset_name")
            or os.path.basename(os.path.normpath(data_path)),
            "total_time": round(total_time, 1),
            "best_val_loss": round(best_val_loss, 6),
            "best_checkpoint_metric": checkpoint_metric,
            "best_checkpoint_score": round(best_checkpoint_score, 6),
            "best_epoch": best_epoch,
            "accuracy": round(accuracy * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "classification_report": per_class,
            "model_path": save_path,
            "num_classes": num_classes,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "roc_curves": roc_curves,
            "common_fpr": common_fpr.tolist(),
            "epoch_history": epoch_history,
            "runtime": {
                "device": device.type,
                "dataset_name": config.get("dataset_name")
                or os.path.basename(os.path.normpath(data_path)),
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "mixed_precision": use_amp,
                "optimizer": config.get("optimizer_name", "AdamW"),
                "scheduler": "ReduceLROnPlateau",
                "early_stopping": early_stopping,
                "split_strategy": split_strategy,
                "checkpoint_metric": checkpoint_metric,
                "sampler_strategy": sampler_strategy,
                "loss_name": loss_name,
                "class_weight_strategy": class_weight_strategy,
                "label_smoothing": label_smoothing,
                "focal_gamma": focal_gamma,
                "effective_number_beta": effective_number_beta,
                "augmentation_profile": augmentation_profile,
                "seed": seed,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "fine_tune_learning_rate": fine_tune_learning_rate,
                "input_size": cfg["input_size"],
            },
            "efficiency": {
                "train_images_seen": train_images_seen,
                "train_images_per_second": round(train_images_seen / total_time, 4)
                if total_time > 0
                else 0.0,
                "test_images": len(test_ds),
                "test_eval_seconds": round(eval_time, 4),
                "test_images_per_second": round(len(test_ds) / eval_time, 4)
                if eval_time > 0
                else 0.0,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "model_size_mb": round(model_size_mb, 4),
            },
        }

        history_path = os.path.join(MODELS_SAVE_DIR, "training_history.json")
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "experiment_name": experiment_name,
            "dataset_name": os.path.basename(os.path.normpath(data_path)),
            "config": config,
            "result": result
        }
        
        try:
            history = []
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            history.insert(0, history_entry)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar histórico de treinamento: {e}")

        self.last_result = result
        callback(result)


# Instância singleton
training_manager = TrainingManager()
