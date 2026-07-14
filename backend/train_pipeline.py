r"""
Pipeline de treinamento em linha de comando.

Edite backend/training_jobs.toml para definir os treinamentos desejados e então rode:

    .\.venv\Scripts\python.exe -m backend.train_pipeline

Filtros opcionais:

    .\.venv\Scripts\python.exe -m backend.train_pipeline --id mobilenetv3_sem_fundo_baseline
    .\.venv\Scripts\python.exe -m backend.train_pipeline --dataset sem_fundo --model MobileNetV3
    .\.venv\Scripts\python.exe -m backend.train_pipeline --tag modesta
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only needed on Python < 3.11
    tomllib = None

from backend.services.training_service import (
    TRAINING_MODEL_CONFIGS,
    WORKSPACE_ROOT,
    training_manager,
)


DATA_ROOT = Path(WORKSPACE_ROOT) / "data"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("training_jobs.toml")
PIPELINE_RUNS_DIR = Path(WORKSPACE_ROOT) / "models" / "pipeline_runs"
COMPLETED_STATUS = "success"
PENDING_STATUS = "pending"
RUNNING_STATUS = "running"
ERROR_STATUS = "error"

REPORT_FRONT_MATTER_KEYS = [
    "id",
    "enabled",
    "tags",
    "notes",
    "dataset_name",
    "data_path",
    "model_name",
    "experiment_name",
]


def _safe_name(value: str) -> str:
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    )
    return "_".join(part for part in safe.split("_") if part)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa jobs de treinamento definidos em TOML."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Arquivo TOML com a lista [[jobs]]. Padrao: backend/training_jobs.toml.",
    )
    parser.add_argument(
        "--run-file",
        help=(
            "Arquivo JSON de execucao criado anteriormente em models/pipeline_runs/. "
            "Use para retomar a mesma fila."
        ),
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Reexecuta jobs ja marcados como success no arquivo de execucao.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Inclui jobs com enabled = false.",
    )
    parser.add_argument("--id", dest="ids", action="append", help="Filtra por id do job.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Filtra por dataset_name.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Filtra por model_name.",
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        help="Filtra por experiment_name.",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        help="Filtra jobs que tenham a tag informada. Pode repetir.",
    )
    return parser.parse_args()


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "Esta versao do Python nao tem tomllib. Use Python 3.11+ "
            "ou instale uma alternativa e adapte o carregamento do TOML."
        )
    with path.open("rb") as file:
        return tomllib.load(file)


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"tags deve ser string ou lista de strings. Recebido: {value!r}")


def _load_jobs(config_path: Path, include_disabled: bool) -> list[dict[str, Any]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")

    config = _load_toml(config_path)
    raw_jobs = config.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError("O arquivo TOML deve conter uma lista [[jobs]].")

    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_job in enumerate(raw_jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Job #{index} deve ser um bloco TOML.")

        job = dict(raw_job)
        job_id = str(job.get("id", "")).strip()
        if not job_id:
            raise ValueError(f"Job #{index} nao tem id.")
        if job_id in seen_ids:
            raise ValueError(f"Job duplicado no TOML: {job_id}")
        seen_ids.add(job_id)

        job["id"] = job_id
        job["enabled"] = bool(job.get("enabled", True))
        job["tags"] = _normalize_tags(job.get("tags", []))
        job["notes"] = str(job.get("notes", "")).strip()

        if include_disabled or job["enabled"]:
            jobs.append(job)

    return jobs


def _matches_any(value: Any, filters: list[str] | None) -> bool:
    if not filters:
        return True
    return str(value) in set(filters)


def _job_matches_filters(job: dict[str, Any], args: argparse.Namespace) -> bool:
    if not _matches_any(job.get("id"), args.ids):
        return False
    if not _matches_any(job.get("dataset_name"), args.datasets):
        return False
    if not _matches_any(job.get("model_name"), args.models):
        return False
    if not _matches_any(job.get("experiment_name", "default"), args.experiments):
        return False
    if args.tags:
        job_tags = set(_normalize_tags(job.get("tags", [])))
        if not set(args.tags).issubset(job_tags):
            return False
    return True


def _filter_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return [job for job in jobs if _job_matches_filters(job, args)]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_run_state(run_state: dict[str, Any], run_file: Path) -> None:
    run_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_text(
        json.dumps(run_state, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _build_run_file_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PIPELINE_RUNS_DIR / f"pipeline_run_{timestamp}.json"


def _active_filters(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "ids": args.ids or [],
        "datasets": args.datasets or [],
        "models": args.models or [],
        "experiments": args.experiments or [],
        "tags": args.tags or [],
    }


def _create_run_state(
    jobs: list[dict[str, Any]],
    config_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "created_at": now,
        "updated_at": now,
        "config_path": str(config_path),
        "filters": _active_filters(args),
        "include_disabled": bool(args.include_disabled),
        "jobs": [
            {
                "id": job["id"],
                "status": PENDING_STATUS,
                "attempts": 0,
                "started_at": None,
                "finished_at": None,
                "job": job,
            }
            for job in jobs
        ],
    }


def _load_run_state(run_file: Path) -> dict[str, Any]:
    return json.loads(run_file.read_text(encoding="utf-8"))


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _append_job_parameters(lines: list[str], job: dict[str, Any]) -> None:
    for key in REPORT_FRONT_MATTER_KEYS:
        if key in job:
            lines.append(f"{key}: {_format_value(job[key])}")

    remaining_keys = sorted(key for key in job if key not in REPORT_FRONT_MATTER_KEYS)
    for key in remaining_keys:
        lines.append(f"{key}: {_format_value(job[key])}")


def _validate_job(job: dict[str, Any]) -> dict[str, Any]:
    split_strategy = job.get("split_strategy", "random")
    required_fields = [
        "id",
        "model_name",
        "data_path",
        "dataset_name",
        "num_epochs",
        "learning_rate",
        "patience",
    ]
    if split_strategy != "predefined":
        required_fields.extend(["train_split", "val_split"])

    missing = [field for field in required_fields if field not in job]
    if missing:
        raise ValueError(f"Job '{job.get('id', 'sem_id')}' sem campos: {', '.join(missing)}")

    model_name = job["model_name"]
    if model_name not in TRAINING_MODEL_CONFIGS:
        models = ", ".join(TRAINING_MODEL_CONFIGS)
        raise ValueError(f"Modelo inválido: '{model_name}'. Opções: {models}")

    configured_data_path = Path(str(job["data_path"]))
    if not configured_data_path.is_absolute():
        configured_data_path = Path(WORKSPACE_ROOT) / configured_data_path
    data_path = configured_data_path.resolve()
    allowed_root = DATA_ROOT.resolve()
    if allowed_root not in data_path.parents and data_path != allowed_root:
        raise ValueError(
            f"Dataset fora de data/: '{data_path}'. Use um diretório dentro de '{allowed_root}'."
        )
    if not data_path.is_dir():
        raise ValueError(f"Dataset não encontrado: '{data_path}'.")

    if split_strategy not in {"random", "stratified", "predefined"}:
        raise ValueError(
            f"split_strategy inválido para '{model_name}': {split_strategy}."
        )

    if split_strategy != "predefined":
        train_split = float(job.get("train_split", 0.8))
        val_split = float(job.get("val_split", 0.1))
        if train_split + val_split >= 0.95:
            raise ValueError(
                f"Splits inválidos para '{model_name}': train_split + val_split deve ser < 0.95."
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

    loss_name = job.get("loss_name", "cross_entropy")
    if loss_name not in {"cross_entropy", "focal"}:
        raise ValueError(
            f"loss_name inválido para '{model_name}': {loss_name}."
        )

    class_weight_strategy = job.get("class_weight_strategy", "sqrt_inverse")
    if class_weight_strategy not in {
        "sqrt_inverse",
        "inverse",
        "effective_number",
        "none",
    }:
        raise ValueError(
            "class_weight_strategy inválido para "
            f"'{model_name}': {class_weight_strategy}."
        )

    label_smoothing = float(job.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            f"label_smoothing inválido para '{model_name}': {label_smoothing}."
        )

    focal_gamma = float(job.get("focal_gamma", 1.5))
    if focal_gamma < 0.0:
        raise ValueError(f"focal_gamma inválido para '{model_name}': {focal_gamma}.")

    effective_number_beta = float(job.get("effective_number_beta", 0.999))
    if not 0.0 < effective_number_beta < 1.0:
        raise ValueError(
            "effective_number_beta inválido para "
            f"'{model_name}': {effective_number_beta}."
        )

    augmentation_profile = job.get("augmentation_profile", "standard")
    if augmentation_profile not in {
        "standard",
        "conservative_color",
        "no_color_jitter",
    }:
        raise ValueError(
            f"augmentation_profile inválido para '{model_name}': {augmentation_profile}."
        )

    normalized = dict(job)
    normalized["data_path"] = str(data_path)
    normalized["dataset_name"] = normalized.get("dataset_name") or data_path.name
    return normalized


def _print_job_header(index: int, total: int, job: dict[str, Any]) -> None:
    batch_size = job.get("batch_size", "model-default")
    print()
    print("=" * 72)
    print(
        f"[{index}/{total}] {job['id']} | {job['model_name']} | "
        f"experiment={job.get('experiment_name', 'default')} | "
        f"dataset={job.get('dataset_name', Path(job['data_path']).name)} | "
        f"epochs={job['num_epochs']} | batch={batch_size}"
    )
    if job.get("notes"):
        print(f"notes={job['notes']}")
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
    _append_job_parameters(lines, job)
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
        lines.append(f"loss_name: {runtime.get('loss_name', 'unknown')}")
        lines.append(
            f"class_weight_strategy: {runtime.get('class_weight_strategy', 'unknown')}"
        )
        lines.append(f"label_smoothing: {runtime.get('label_smoothing', 'unknown')}")
        lines.append(f"focal_gamma: {runtime.get('focal_gamma', 'unknown')}")
        lines.append(f"augmentation_profile: {runtime.get('augmentation_profile', 'unknown')}")
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
    model_name = str(job.get("model_name", "modelo_desconhecido")).lower()
    dataset_name = _safe_name(
        str(job.get("dataset_name") or Path(str(job.get("data_path", "dataset"))).name)
    )
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
    lines.append(f"Modelo: {job.get('model_name', 'unknown')}")
    lines.append(f"Experimento: {job.get('experiment_name', 'default')}")
    lines.append(
        f"Dataset nome: {job.get('dataset_name', Path(str(job.get('data_path', 'dataset'))).name)}"
    )
    lines.append(f"Dataset: {job.get('data_path', 'unknown')}")
    lines.append("")
    lines.append("CONFIGURACAO")
    lines.append("-" * 72)
    _append_job_parameters(lines, job)
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
            f"{entry.get('id', 'sem_id')} | {entry['model_name']} | status={entry['status']} | "
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
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(WORKSPACE_ROOT) / config_path).resolve()

    if args.run_file:
        run_file = Path(args.run_file)
        if not run_file.is_absolute():
            run_file = (Path(WORKSPACE_ROOT) / run_file).resolve()
        run_state = _load_run_state(run_file)
        run_entries = run_state.get("jobs", [])
        print(f"Retomando execucao registrada em: {run_file}")
    else:
        all_jobs = _load_jobs(config_path, args.include_disabled)
        selected_jobs = _filter_jobs(all_jobs, args)
        if not selected_jobs:
            print("Nenhum job encontrado para os filtros informados.")
            return 1
        run_file = _build_run_file_path()
        run_state = _create_run_state(selected_jobs, config_path, args)
        run_entries = run_state["jobs"]
        _write_run_state(run_state, run_file)
        print(f"Arquivo de execucao criado: {run_file}")

    runnable_entries = [
        entry for entry in run_entries
        if args.rerun_completed or entry.get("status") != COMPLETED_STATUS
    ]
    if not runnable_entries:
        print("Nenhum job pendente. Use --rerun-completed para reexecutar sucessos.")
        return 0

    print(f"Executando {len(runnable_entries)} job(s) de treinamento.")
    started_at = time.time()
    failed_jobs = 0
    summary_entries = []

    for index, entry in enumerate(runnable_entries, start=1):
        job = dict(entry["job"])
        entry["status"] = RUNNING_STATUS
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["started_at"] = datetime.now().isoformat(timespec="seconds")
        entry["finished_at"] = None
        entry["error"] = None
        _write_run_state(run_state, run_file)

        try:
            job = _validate_job(job)
            entry["job"] = job
            _write_run_state(run_state, run_file)
            _print_job_header(index, len(runnable_entries), job)
            callback = _build_progress_callback(job)
            training_manager.run_blocking(job, callback)
            result = training_manager.last_result
            if result and result.get("type") == "training_complete":
                report_path = _write_training_report(job, result)
                print(f"[report] {report_path}")
                entry["status"] = COMPLETED_STATUS
                entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
                entry["model_path"] = result.get("model_path", "")
                entry["report_path"] = str(report_path)
                entry["result"] = {
                    "accuracy": result.get("accuracy", 0.0),
                    "macro_f1": result.get("macro_f1", 0.0),
                    "best_val_loss": result.get("best_val_loss", 0.0),
                    "best_checkpoint_metric": result.get("best_checkpoint_metric", "val_loss"),
                    "best_checkpoint_score": result.get("best_checkpoint_score", 0.0),
                    "total_time": result.get("total_time", 0.0),
                }
                _write_run_state(run_state, run_file)
                summary_entries.append({
                    "id": job["id"],
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
            else:
                failed_jobs += 1
                error_message = "Treinamento finalizou sem resultado training_complete."
                report_path = _write_error_report(job, error_message)
                entry["status"] = ERROR_STATUS
                entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
                entry["error"] = error_message
                entry["report_path"] = str(report_path)
                _write_run_state(run_state, run_file)
                print(f"[error] Job {index} falhou: {error_message}")
                print(f"[report] {report_path}")
                summary_entries.append({
                    "id": job["id"],
                    "model_name": job["model_name"],
                    "experiment_name": job.get("experiment_name", "default"),
                    "dataset_name": job.get("dataset_name", Path(job["data_path"]).name),
                    "data_path": job["data_path"],
                    "status": "error",
                    "error": error_message,
                    "report_path": str(report_path),
                })
        except KeyboardInterrupt:
            print("\n[interrupt] cancelado pelo usuário.")
            entry["status"] = PENDING_STATUS
            entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
            entry["error"] = "Interrompido pelo usuario."
            _write_run_state(run_state, run_file)
            return 130
        except Exception as exc:
            failed_jobs += 1
            report_path = _write_error_report(job, str(exc))
            entry["status"] = ERROR_STATUS
            entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
            entry["error"] = str(exc)
            entry["report_path"] = str(report_path)
            _write_run_state(run_state, run_file)
            print(f"[error] Job {index} falhou: {exc}")
            print(f"[report] {report_path}")
            summary_entries.append({
                "id": job["id"],
                "model_name": job.get("model_name", "unknown"),
                "experiment_name": job.get("experiment_name", "default"),
                "dataset_name": job.get(
                    "dataset_name", Path(str(job.get("data_path", "dataset"))).name
                ),
                "data_path": job.get("data_path", "unknown"),
                "status": "error",
                "error": str(exc),
                "report_path": str(report_path),
            })
            continue

    total_time = time.time() - started_at
    summary_path = _write_pipeline_summary(summary_entries)
    print()
    print(f"Pipeline concluído em {total_time:.1f}s. Falhas: {failed_jobs}.")
    print(f"[run] {run_file}")
    print(f"[summary] {summary_path}")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
