"""
TrainingManager — Serviço centralizado de treinamento de modelos.

Executa o treinamento em thread separada, reportando progresso
via callback (consumido pelo WebSocket).
"""

import os
import time
import threading
import json
from datetime import datetime
from typing import Callable, Optional

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except RuntimeError:
            pass  # já configurado

    num_workers = (
        min(4, max(1, cpu_threads // 4))
        if device.type == "cpu"
        else min(8, cpu_threads)
    )
    pin_memory = device.type == "cuda"
    persistent_workers = device.type == "cuda" and num_workers > 0
    prefetch_factor = 1 if num_workers > 0 else 2

    return device, num_workers, pin_memory, persistent_workers, prefetch_factor


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


def _build_train_transforms(input_size: int):
    """Transforms agressivos exclusivos para treinamento (lida com desbalanceamento)."""
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _build_val_transforms(input_size: int):
    """Transforms rígidos sem distorções para Teste e Validação."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


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
        data_path = config["data_path"]
        batch_size = config.get("batch_size", 16)
        num_epochs = config.get("num_epochs", 20)
        learning_rate = config.get("learning_rate", 1e-4)
        patience = config.get("patience", 5)
        train_split = config.get("train_split", 0.8)
        val_split = config.get("val_split", 0.1)

        if model_name not in TRAINING_MODEL_CONFIGS:
            raise ValueError(f"Modelo '{model_name}' não suportado.")

        cfg = TRAINING_MODEL_CONFIGS[model_name]

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

        train_ds_raw, val_ds_raw, test_ds_raw = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_transforms = _build_train_transforms(cfg["input_size"])
        val_transforms = _build_val_transforms(cfg["input_size"])

        train_ds = TransformSubset(train_ds_raw, transform=train_transforms)
        val_ds = TransformSubset(val_ds_raw, transform=val_transforms)
        test_ds = TransformSubset(test_ds_raw, transform=val_transforms)

        # ── Class Weights ──
        callback({"type": "status", "message": "Calculando os pesos das classes..."})
        train_indices = train_ds_raw.indices
        targets = [dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets, minlength=num_classes)
        
        # Suaviza para evitar explodir em classes quase vazias: 1 / sqrt(N)
        weights = 1.0 / np.sqrt(class_counts + 1e-8)
        weights = (weights / np.sum(weights)) * num_classes
        class_weights = torch.FloatTensor(weights).to(device)

        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

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
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # ── Training loop ──
        best_val_loss = float("inf")
        epochs_no_improve = 0
        total_batches = len(train_loader)
        start_time = time.time()

        # Caminho para salvar pesos com timestamp
        os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            MODELS_SAVE_DIR, f"soybean_model_{model_name.lower()}_{timestamp_str}.pth"
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

            model.train()
            running_loss = 0.0

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

                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(data)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

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
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            epoch_val_loss = val_loss / len(val_loader)
            elapsed = time.time() - start_time

            callback({
                "type": "epoch_complete",
                "epoch": epoch + 1,
                "total_epochs": num_epochs,
                "train_loss": round(epoch_train_loss, 6),
                "val_loss": round(epoch_val_loss, 6),
                "elapsed_seconds": round(elapsed, 1),
            })

            # ── Early stopping ──
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes
                }
                torch.save(checkpoint, save_path)
                has_saved_checkpoint = True
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    callback({
                        "type": "status",
                        "message": f"Early stopping na epoch {epoch + 1}",
                    })
                    break

        # ── Avaliação no test set ──
        callback({"type": "status", "message": "Avaliando modelo no conjunto de teste..."})

        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels_dev = labels.to(device, non_blocking=pin_memory)
                outputs = model(images)
                
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True,
            digits=4,
        )

        # Accuracy
        accuracy = report.get("accuracy", 0.0)

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

        result = {
            "type": "training_complete",
            "total_time": round(total_time, 1),
            "best_val_loss": round(best_val_loss, 6),
            "accuracy": round(accuracy * 100, 2),
            "classification_report": per_class,
            "model_path": save_path,
            "num_classes": num_classes,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "roc_curves": roc_curves,
            "common_fpr": common_fpr.tolist(),
        }

        history_path = os.path.join(MODELS_SAVE_DIR, "training_history.json")
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
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
