r"""
Pipeline de treinamento em linha de comando.

Edite a lista PIPELINE abaixo para definir os treinamentos desejados e então rode:

    .\.venv\Scripts\python.exe -m backend.train_pipeline
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.services.training_service import (
    TRAINING_MODEL_CONFIGS,
    WORKSPACE_ROOT,
    training_manager,
)


DATA_ROOT = Path(WORKSPACE_ROOT) / "data"
DATASET_NAMES = [
    "sem_fundo",
    "com_fundo",
]

COMMON_CONFIG: dict[str, Any] = {
    "num_epochs": 20,
    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,
    "optimizer_name": "AdamW",
    "weight_decay": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_min_lr": 1e-6,
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "MobileNetV3": {
        "batch_size": 8,
        "learning_rate": 1.5e-4,
        "fine_tune_learning_rate": 1e-4,
        "accumulation_steps": 1,
        "freeze_backbone_epochs": 2,
    },
    "EfficientNetB0": {
        "batch_size": 6,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 8e-5,
        "accumulation_steps": 1,
        "freeze_backbone_epochs": 2,
    },
    "EfficientNetB2": {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 8e-5,
        "accumulation_steps": 1,
        "freeze_backbone_epochs": 2,
    },
    "EfficientNetB3": {
        "batch_size": 2,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 7e-5,
        "accumulation_steps": 4,
        "freeze_backbone_epochs": 2,
    },
    # Disponivel para reativar, mas fora da comparacao padrao por custo alto
    # para ganho pequeno nos relatorios atuais.
    "ResNet50": {
        "batch_size": 2,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 8e-5,
        "accumulation_steps": 2,
        "freeze_backbone_epochs": 2,
    },
    # Fora da comparacao padrao: relatorios anteriores indicaram baixo
    # desempenho e tempo de treinamento muito alto.
    "EfficientNetB7": {
        "batch_size": 1,
        "num_epochs": 16,
        "learning_rate": 1e-4,
        "fine_tune_learning_rate": 5e-5,
        "accumulation_steps": 8,
        "freeze_backbone_epochs": 3,
    },
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "early_stopping": True,
        "split_strategy": "random",
        "checkpoint_metric": "val_loss",
        "sampler_strategy": "shuffle",
        "patience": 5,
    },
    "experimental": {
        "early_stopping": False,
        "split_strategy": "stratified",
        "checkpoint_metric": "val_macro_f1",
        "sampler_strategy": "weighted",
        "patience": 5,
    },
}

# Relatorios atuais justificam comparar estes modelos:
# - MobileNetV3: melhor macro F1 e menor tempo entre os melhores.
# - EfficientNetB0: rapido e competitivo.
# - EfficientNetB2: melhor acuracia entre EfficientNets testados.
# - EfficientNetB3: novo ponto intermediario entre B2 e B7.
CANDIDATE_MODELS = [
    "MobileNetV3",
    "EfficientNetB0",
    "EfficientNetB2",
    "EfficientNetB3",
]


def _build_pipeline() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset_name in DATASET_NAMES:
        for model_name in CANDIDATE_MODELS:
            model_config = MODEL_CONFIGS[model_name]
            for experiment_name, experiment_config in EXPERIMENTS.items():
                job = {
                    **COMMON_CONFIG,
                    **model_config,
                    **experiment_config,
                    "model_name": model_name,
                    "experiment_name": experiment_name,
                    "dataset_name": dataset_name,
                    "data_path": str(DATA_ROOT / dataset_name),
                }
                jobs.append(job)
    return jobs


PIPELINE: list[dict[str, Any]] = _build_pipeline()


def _safe_name(value: str) -> str:
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    )
    return "_".join(part for part in safe.split("_") if part)


def _validate_job(job: dict[str, Any]) -> dict[str, Any]:
    model_name = job["model_name"]
    if model_name not in TRAINING_MODEL_CONFIGS:
        models = ", ".join(TRAINING_MODEL_CONFIGS)
        raise ValueError(f"Modelo inválido: '{model_name}'. Opções: {models}")

    data_path = Path(job["data_path"]).resolve()
    allowed_root = DATA_ROOT.resolve()
    if allowed_root not in data_path.parents and data_path != allowed_root:
        raise ValueError(
            f"Dataset fora de data/: '{data_path}'. Use um diretório dentro de '{allowed_root}'."
        )
    if not data_path.is_dir():
        raise ValueError(f"Dataset não encontrado: '{data_path}'.")

    train_split = float(job.get("train_split", 0.8))
    val_split = float(job.get("val_split", 0.1))
    if train_split + val_split >= 0.95:
        raise ValueError(
            f"Splits inválidos para '{model_name}': train_split + val_split deve ser < 0.95."
        )

    split_strategy = job.get("split_strategy", "random")
    if split_strategy not in {"random", "stratified"}:
        raise ValueError(
            f"split_strategy inválido para '{model_name}': {split_strategy}."
        )

    checkpoint_metric = job.get("checkpoint_metric", "val_loss")
    if checkpoint_metric not in {"val_loss", "val_accuracy", "val_macro_f1"}:
        raise ValueError(
            f"checkpoint_metric inválido para '{model_name}': {checkpoint_metric}."
        )

    sampler_strategy = job.get("sampler_strategy", "shuffle")
    if sampler_strategy not in {"shuffle", "weighted"}:
        raise ValueError(
            f"sampler_strategy inválido para '{model_name}': {sampler_strategy}."
        )

    normalized = dict(job)
    normalized["data_path"] = str(data_path)
    normalized["dataset_name"] = normalized.get("dataset_name") or data_path.name
    return normalized


def _print_job_header(index: int, total: int, job: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(
        f"[{index}/{total}] {job['model_name']} | "
        f"experiment={job.get('experiment_name', 'default')} | "
        f"dataset={job.get('dataset_name', Path(job['data_path']).name)} | "
        f"epochs={job['num_epochs']} | batch={job['batch_size']}"
    )
    print("=" * 72)


def _build_progress_callback(job: dict[str, Any]):
    last_batch_line = {"printed": False}

    def callback(message: dict[str, Any]) -> None:
        msg_type = message.get("type")

        if msg_type == "status":
            if last_batch_line["printed"]:
                print()
                last_batch_line["printed"] = False
            print(f"[status] {message['message']}")
            return

        if msg_type == "batch_progress":
            epoch = message["epoch"]
            total_epochs = message["total_epochs"]
            batch = message["batch"]
            total_batches = message["total_batches"]
            loss = message["loss"]
            progress = (batch / total_batches) * 100 if total_batches else 100.0
            line = (
                f"\r[batch] {job['model_name']} | "
                f"{job.get('experiment_name', 'default')} | "
                f"epoch {epoch}/{total_epochs} | "
                f"batch {batch}/{total_batches} | "
                f"{progress:5.1f}% | loss={loss:.4f}"
            )
            print(line, end="", flush=True)
            last_batch_line["printed"] = True
            return

        if last_batch_line["printed"]:
            print()
            last_batch_line["printed"] = False

        if msg_type == "epoch_complete":
            print(
                f"[epoch] {message['epoch']}/{message['total_epochs']} | "
                f"train_loss={message['train_loss']:.6f} | "
                f"val_loss={message['val_loss']:.6f} | "
                f"val_macro_f1={message.get('val_macro_f1', 0.0):.2f}% | "
                f"checkpoint={message.get('checkpoint_metric', 'val_loss')}:"
                f"{message.get('checkpoint_score', 0.0):.6f} | "
                f"elapsed={message['elapsed_seconds']:.1f}s"
            )
            return

        if msg_type == "training_complete":
            print(
                f"[done] accuracy={message['accuracy']:.2f}% | "
                f"macro_f1={message.get('macro_f1', 0.0):.2f}% | "
                f"best_{message.get('best_checkpoint_metric', 'val_loss')}="
                f"{message.get('best_checkpoint_score', 0.0):.6f}"
            )
            print(f"[model] {message['model_path']}")
            return

        if msg_type == "training_cancelled":
            print("[cancelled] treinamento interrompido.")
            return

        if msg_type == "training_error":
            print(f"[error] {message['message']}")
            return

        print(f"[event] {message}")

    return callback


def _write_training_report(job: dict[str, Any], result: dict[str, Any]) -> Path:
    model_path = Path(result["model_path"])
    report_path = model_path.with_suffix(".txt")

    lines: list[str] = []
    lines.append("RELATORIO DE TREINAMENTO")
    lines.append("=" * 72)
    lines.append(f"Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Modelo: {job['model_name']}")
    lines.append(f"Experimento: {job.get('experiment_name', 'default')}")
    lines.append(f"Dataset nome: {job.get('dataset_name', Path(job['data_path']).name)}")
    lines.append(f"Dataset: {job['data_path']}")
    lines.append(f"Arquivo de pesos: {result['model_path']}")
    lines.append("")
    lines.append("CONFIGURACAO")
    lines.append("-" * 72)
    lines.append(f"batch_size: {job['batch_size']}")
    lines.append(f"num_epochs: {job['num_epochs']}")
    lines.append(f"learning_rate: {job['learning_rate']}")
    lines.append(f"fine_tune_learning_rate: {job.get('fine_tune_learning_rate', job['learning_rate'])}")
    lines.append(f"early_stopping: {job.get('early_stopping', True)}")
    lines.append(f"split_strategy: {job.get('split_strategy', 'random')}")
    lines.append(f"checkpoint_metric: {job.get('checkpoint_metric', 'val_loss')}")
    lines.append(f"sampler_strategy: {job.get('sampler_strategy', 'shuffle')}")
    lines.append(f"patience: {job['patience']}")
    lines.append(f"train_split: {job['train_split']}")
    lines.append(f"val_split: {job['val_split']}")
    lines.append(f"seed: {job.get('seed', 42)}")
    lines.append(f"optimizer_name: {job.get('optimizer_name', 'AdamW')}")
    lines.append(f"weight_decay: {job.get('weight_decay', 1e-4)}")
    lines.append(f"scheduler_factor: {job.get('scheduler_factor', 0.5)}")
    lines.append(f"scheduler_patience: {job.get('scheduler_patience', 2)}")
    lines.append(f"scheduler_min_lr: {job.get('scheduler_min_lr', 1e-6)}")
    lines.append(f"accumulation_steps: {job.get('accumulation_steps', 1)}")
    lines.append(f"freeze_backbone_epochs: {job.get('freeze_backbone_epochs', 0)}")
    lines.append("")
    lines.append("RESULTADO FINAL")
    lines.append("-" * 72)
    lines.append(f"accuracy: {result.get('accuracy', 0.0):.2f}%")
    lines.append(f"macro_f1: {result.get('macro_f1', 0.0):.2f}%")
    lines.append(f"best_val_loss: {result.get('best_val_loss', 0.0):.6f}")
    lines.append(f"best_checkpoint_metric: {result.get('best_checkpoint_metric', 'val_loss')}")
    lines.append(f"best_checkpoint_score: {result.get('best_checkpoint_score', 0.0):.6f}")
    lines.append(f"best_epoch: {result.get('best_epoch', 0)}")
    lines.append(f"total_time: {result.get('total_time', 0.0):.1f}s")
    lines.append(f"num_classes: {result.get('num_classes', 0)}")
    lines.append(f"class_names: {', '.join(result.get('class_names', []))}")

    runtime = result.get("runtime", {})
    if runtime:
        lines.append("")
        lines.append("RUNTIME")
        lines.append("-" * 72)
        lines.append(f"dataset_name: {runtime.get('dataset_name', job.get('dataset_name', 'unknown'))}")
        lines.append(f"device: {runtime.get('device', 'unknown')}")
        lines.append(f"num_workers: {runtime.get('num_workers', 'unknown')}")
        lines.append(f"pin_memory: {runtime.get('pin_memory', 'unknown')}")
        lines.append(f"mixed_precision: {runtime.get('mixed_precision', 'unknown')}")
        lines.append(f"optimizer: {runtime.get('optimizer', 'unknown')}")
        lines.append(f"scheduler: {runtime.get('scheduler', 'unknown')}")
        lines.append(f"split_strategy: {runtime.get('split_strategy', 'unknown')}")
        lines.append(f"checkpoint_metric: {runtime.get('checkpoint_metric', 'unknown')}")
        lines.append(f"sampler_strategy: {runtime.get('sampler_strategy', 'unknown')}")
        lines.append(f"input_size: {runtime.get('input_size', 'unknown')}")
        lines.append(f"effective_batch_size: {runtime.get('effective_batch_size', 'unknown')}")

    efficiency = result.get("efficiency", {})
    if efficiency:
        lines.append("")
        lines.append("EFICIENCIA")
        lines.append("-" * 72)
        lines.append(f"train_images_seen: {efficiency.get('train_images_seen', 0)}")
        lines.append(
            f"train_images_per_second: {efficiency.get('train_images_per_second', 0.0):.4f}"
        )
        lines.append(f"test_images: {efficiency.get('test_images', 0)}")
        lines.append(f"test_eval_seconds: {efficiency.get('test_eval_seconds', 0.0):.4f}")
        lines.append(
            f"test_images_per_second: {efficiency.get('test_images_per_second', 0.0):.4f}"
        )
        lines.append(f"parameter_count: {efficiency.get('parameter_count', 0)}")
        lines.append(
            f"trainable_parameter_count: {efficiency.get('trainable_parameter_count', 0)}"
        )
        lines.append(f"model_size_mb: {efficiency.get('model_size_mb', 0.0):.4f}")

    lines.append("")
    lines.append("METRICAS POR CLASSE")
    lines.append("-" * 72)

    for item in result.get("classification_report", []):
        lines.append(
            f"{item['class']}: "
            f"precision={item['precision']:.2f}% | "
            f"recall={item['recall']:.2f}% | "
            f"f1={item['f1']:.2f}% | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("MATRIZ DE CONFUSAO")
    lines.append("-" * 72)
    for row in result.get("confusion_matrix", []):
        lines.append(" ".join(str(value) for value in row))

    if result.get("roc_curves"):
        lines.append("")
        lines.append("ROC AUC POR CLASSE")
        lines.append("-" * 72)
        for curve in result["roc_curves"]:
            lines.append(f"{curve['class']}: auc={curve['auc']:.4f}")

    if result.get("epoch_history"):
        lines.append("")
        lines.append("HISTORICO POR EPOCA")
        lines.append("-" * 72)
        for item in result["epoch_history"]:
            lines.append(
                f"epoch={item['epoch']} | phase={item['phase']} | "
                f"train_loss={item['train_loss']:.6f} | "
                f"val_loss={item['val_loss']:.6f} | "
                f"val_accuracy={item.get('val_accuracy', 0.0):.4f}% | "
                f"val_macro_f1={item.get('val_macro_f1', 0.0):.4f}% | "
                f"checkpoint_score={item.get('checkpoint_score', 0.0):.6f} | "
                f"lr={item['learning_rate']:.8f} | "
                f"elapsed={item['elapsed_seconds']:.1f}s"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _build_error_report_path(job: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = job["model_name"].lower()
    dataset_name = _safe_name(job.get("dataset_name", Path(job["data_path"]).name))
    experiment_name = _safe_name(job.get("experiment_name", "default"))
    return (
        Path(WORKSPACE_ROOT)
        / "models"
        / f"soybean_model_{model_name}_{dataset_name}_{experiment_name}_{timestamp}_error.txt"
    )


def _write_error_report(job: dict[str, Any], error_message: str) -> Path:
    report_path = _build_error_report_path(job)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("RELATORIO DE ERRO DE TREINAMENTO")
    lines.append("=" * 72)
    lines.append(f"Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Modelo: {job['model_name']}")
    lines.append(f"Experimento: {job.get('experiment_name', 'default')}")
    lines.append(f"Dataset nome: {job.get('dataset_name', Path(job['data_path']).name)}")
    lines.append(f"Dataset: {job['data_path']}")
    lines.append("")
    lines.append("CONFIGURACAO")
    lines.append("-" * 72)
    lines.append(f"batch_size: {job['batch_size']}")
    lines.append(f"num_epochs: {job['num_epochs']}")
    lines.append(f"learning_rate: {job['learning_rate']}")
    lines.append(f"fine_tune_learning_rate: {job.get('fine_tune_learning_rate', job['learning_rate'])}")
    lines.append(f"early_stopping: {job.get('early_stopping', True)}")
    lines.append(f"split_strategy: {job.get('split_strategy', 'random')}")
    lines.append(f"checkpoint_metric: {job.get('checkpoint_metric', 'val_loss')}")
    lines.append(f"sampler_strategy: {job.get('sampler_strategy', 'shuffle')}")
    lines.append(f"patience: {job['patience']}")
    lines.append(f"train_split: {job['train_split']}")
    lines.append(f"val_split: {job['val_split']}")
    lines.append(f"seed: {job.get('seed', 42)}")
    lines.append(f"optimizer_name: {job.get('optimizer_name', 'AdamW')}")
    lines.append(f"weight_decay: {job.get('weight_decay', 1e-4)}")
    lines.append(f"scheduler_factor: {job.get('scheduler_factor', 0.5)}")
    lines.append(f"scheduler_patience: {job.get('scheduler_patience', 2)}")
    lines.append(f"scheduler_min_lr: {job.get('scheduler_min_lr', 1e-6)}")
    lines.append(f"accumulation_steps: {job.get('accumulation_steps', 1)}")
    lines.append(f"freeze_backbone_epochs: {job.get('freeze_backbone_epochs', 0)}")
    lines.append("")
    lines.append("ERRO")
    lines.append("-" * 72)
    lines.append(error_message)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_pipeline_summary(entries: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(WORKSPACE_ROOT) / "models" / f"pipeline_summary_{timestamp}.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("RESUMO DA PIPELINE")
    lines.append("=" * 72)
    lines.append(f"Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for entry in entries:
        lines.append("-" * 72)
        lines.append(
            f"{entry['model_name']} | status={entry['status']} | "
            f"experiment={entry.get('experiment_name', 'default')} | "
            f"dataset={entry.get('dataset_name', Path(entry['data_path']).name)}"
        )
        if entry["status"] == "success":
            lines.append(
                f"accuracy={entry['accuracy']:.2f}% | "
                f"macro_f1={entry['macro_f1']:.2f}% | "
                f"best_val_loss={entry['best_val_loss']:.6f} | "
                f"best_{entry['best_checkpoint_metric']}={entry['best_checkpoint_score']:.6f} | "
                f"total_time={entry['total_time']:.1f}s"
            )
            lines.append(
                f"train_images_per_second={entry['train_images_per_second']:.4f} | "
                f"test_images_per_second={entry['test_images_per_second']:.4f} | "
                f"model_size_mb={entry['model_size_mb']:.4f}"
            )
            lines.append(f"model_path={entry['model_path']}")
            lines.append(f"report_path={entry['report_path']}")
        else:
            lines.append(f"error={entry['error']}")
            lines.append(f"report_path={entry['report_path']}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    if not PIPELINE:
        print("Nenhum job definido em PIPELINE.")
        return 1

    jobs = [_validate_job(job) for job in PIPELINE]
    print(f"Executando {len(jobs)} job(s) de treinamento.")
    started_at = time.time()
    failed_jobs = 0
    summary_entries = []

    for index, job in enumerate(jobs, start=1):
        _print_job_header(index, len(jobs), job)
        callback = _build_progress_callback(job)
        try:
            training_manager.run_blocking(job, callback)
            result = training_manager.last_result
            if result and result.get("type") == "training_complete":
                report_path = _write_training_report(job, result)
                print(f"[report] {report_path}")
                summary_entries.append({
                    "model_name": job["model_name"],
                    "experiment_name": job.get("experiment_name", "default"),
                    "dataset_name": job.get("dataset_name", Path(job["data_path"]).name),
                    "data_path": job["data_path"],
                    "status": "success",
                    "accuracy": result.get("accuracy", 0.0),
                    "macro_f1": result.get("macro_f1", 0.0),
                    "best_val_loss": result.get("best_val_loss", 0.0),
                    "best_checkpoint_metric": result.get("best_checkpoint_metric", "val_loss"),
                    "best_checkpoint_score": result.get("best_checkpoint_score", 0.0),
                    "total_time": result.get("total_time", 0.0),
                    "train_images_per_second": result.get("efficiency", {}).get(
                        "train_images_per_second", 0.0
                    ),
                    "test_images_per_second": result.get("efficiency", {}).get(
                        "test_images_per_second", 0.0
                    ),
                    "model_size_mb": result.get("efficiency", {}).get("model_size_mb", 0.0),
                    "model_path": result.get("model_path", ""),
                    "report_path": str(report_path),
                })
        except KeyboardInterrupt:
            print("\n[interrupt] cancelado pelo usuário.")
            return 130
        except Exception as exc:
            failed_jobs += 1
            report_path = _write_error_report(job, str(exc))
            print(f"[error] Job {index} falhou: {exc}")
            print(f"[report] {report_path}")
            summary_entries.append({
                "model_name": job["model_name"],
                "experiment_name": job.get("experiment_name", "default"),
                "dataset_name": job.get("dataset_name", Path(job["data_path"]).name),
                "data_path": job["data_path"],
                "status": "error",
                "error": str(exc),
                "report_path": str(report_path),
            })
            continue

    total_time = time.time() - started_at
    summary_path = _write_pipeline_summary(summary_entries)
    print()
    print(f"Pipeline concluído em {total_time:.1f}s. Falhas: {failed_jobs}.")
    print(f"[summary] {summary_path}")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
