r"""
Pipeline YOLO de classificacao em linha de comando.

Este script nao altera o TrainingManager existente. Ele cria uma versao splitada
do dataset ImageFolder atual no formato esperado pelo Ultralytics:

    data/_yolo_cls/<dataset>/
      train/<classe>/*.jpg
      val/<classe>/*.jpg
      test/<classe>/*.jpg

Depois rode:

    .\.venv\Scripts\python.exe -m backend.yolo_train_pipeline
"""

from __future__ import annotations

import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.services.training_service import WORKSPACE_ROOT


DATA_ROOT = Path(WORKSPACE_ROOT) / "data"
YOLO_DATA_ROOT = DATA_ROOT / "_yolo_cls"
MODELS_SAVE_DIR = Path(WORKSPACE_ROOT) / "models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


PIPELINE: list[dict[str, Any]] = [
    {
        "model_name": "YOLO26n-cls",
        "weights": "yolo26n-cls.pt",
        "source_data_path": str(DATA_ROOT / "sem_fundo"),
        "batch_size": 16,
        "num_epochs": 30,
        "imgsz": 224,
        "patience": 6,
        "train_split": 0.8,
        "val_split": 0.1,
        "seed": 42,
        "device": 0,  # RTX 3070 via CUDA; use "cpu" se a build local do PyTorch nao tiver CUDA.
        "workers": 0,  # melhor padrao para Windows.
    },
    {
        "model_name": "YOLO26s-cls",
        "weights": "yolo26s-cls.pt",
        "source_data_path": str(DATA_ROOT / "sem_fundo"),
        "batch_size": 12,
        "num_epochs": 30,
        "imgsz": 224,
        "patience": 6,
        "train_split": 0.8,
        "val_split": 0.1,
        "seed": 42,
        "device": 0,
        "workers": 0,
    },
]


def _image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_source_dataset(source_path: Path) -> dict[str, list[Path]]:
    source_path = source_path.resolve()
    allowed_root = DATA_ROOT.resolve()
    if allowed_root not in source_path.parents and source_path != allowed_root:
        raise ValueError(
            f"Dataset fora de data/: '{source_path}'. Use um diretorio dentro de '{allowed_root}'."
        )
    if not source_path.is_dir():
        raise ValueError(f"Dataset nao encontrado: '{source_path}'.")

    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in source_path.iterdir() if path.is_dir()):
        files = _image_files(class_dir)
        if files:
            classes[class_dir.name] = files

    if len(classes) < 2:
        raise ValueError("O dataset precisa ter pelo menos duas classes com imagens.")

    too_small = {name: len(files) for name, files in classes.items() if len(files) < 3}
    if too_small:
        raise ValueError(f"Classes com menos de 3 imagens, insuficiente para train/val/test: {too_small}")

    return classes


def _split_files(files: list[Path], train_split: float, val_split: float, seed: int):
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_size = max(1, int(total * train_split))
    val_size = max(1, int(total * val_split))

    if train_size + val_size >= total:
        train_size = max(1, total - 2)
        val_size = 1

    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    test = shuffled[train_size + val_size :]
    return {"train": train, "val": val, "test": test}


def _copy_split(split: dict[str, list[Path]], target_root: Path, class_name: str) -> dict[str, int]:
    counts = {}
    for split_name, files in split.items():
        class_target = target_root / split_name / class_name
        class_target.mkdir(parents=True, exist_ok=True)
        counts[split_name] = len(files)
        for src in files:
            dst = class_target / src.name
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                dst = class_target / f"{stem}_{abs(hash(str(src))) & 0xffff:x}{suffix}"
            shutil.copy2(src, dst)
    return counts


def _prepare_yolo_classification_dataset(job: dict[str, Any]) -> tuple[Path, dict[str, dict[str, int]]]:
    source_path = Path(job["source_data_path"]).resolve()
    classes = _validate_source_dataset(source_path)

    train_split = float(job.get("train_split", 0.8))
    val_split = float(job.get("val_split", 0.1))
    if train_split + val_split >= 0.95:
        raise ValueError("train_split + val_split deve ser < 0.95 para sobrar teste.")

    seed = int(job.get("seed", 42))
    target_root = YOLO_DATA_ROOT / source_path.name
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    for class_name, files in classes.items():
        split = _split_files(files, train_split, val_split, seed)
        counts[class_name] = _copy_split(split, target_root, class_name)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_data_path": str(source_path),
        "target_root": str(target_root),
        "seed": seed,
        "counts": counts,
    }
    (target_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target_root, counts


def _evaluate_best_model(best_path: Path, dataset_path: Path) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix
    from ultralytics import YOLO

    if not best_path.exists():
        raise FileNotFoundError(f"Peso YOLO nao encontrado para avaliacao: {best_path}")

    test_root = dataset_path / "test"
    class_names = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    y_true = []
    y_pred = []
    inference_times = []
    model = YOLO(str(best_path))

    for class_name in class_names:
        class_dir = test_root / class_name
        for image_path in _image_files(class_dir):
            result = model.predict(str(image_path), verbose=False)[0]
            pred_idx = int(result.probs.top1)
            y_true.append(class_to_idx[class_name])
            y_pred.append(pred_idx)
            inference_times.append(float(result.speed.get("inference", 0.0)))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = []
    for class_name in class_names:
        item = report.get(class_name, {})
        per_class.append({
            "class": class_name,
            "precision": round(float(item.get("precision", 0.0)) * 100, 2),
            "recall": round(float(item.get("recall", 0.0)) * 100, 2),
            "f1": round(float(item.get("f1-score", 0.0)) * 100, 2),
            "support": int(item.get("support", 0)),
        })

    avg_inference_ms = (
        sum(inference_times) / len(inference_times)
        if inference_times
        else 0.0
    )

    return {
        "accuracy": round(float(report.get("accuracy", 0.0)) * 100, 2),
        "classification_report": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "test_images": len(y_true),
        "avg_inference_ms": round(avg_inference_ms, 3),
    }


def _write_report(
    job: dict[str, Any],
    dataset_path: Path,
    counts: dict[str, dict[str, int]],
    result: Any,
    evaluation: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = MODELS_SAVE_DIR / f"soybean_model_{job['model_name'].lower()}_{timestamp}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(getattr(result, "save_dir", ""))
    best_path = save_dir / "weights" / "best.pt" if save_dir else None
    top1 = getattr(result, "top1", None)
    top5 = getattr(result, "top5", None)

    lines = [
        "RELATORIO DE TREINAMENTO YOLO",
        "=" * 72,
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"Modelo: {job['model_name']}",
        f"Pesos iniciais: {job['weights']}",
        f"Dataset YOLO: {dataset_path}",
        f"Best weights: {best_path if best_path and best_path.exists() else 'ver runs/classify'}",
        "",
        "CONFIGURACAO",
        "-" * 72,
        f"batch_size: {job['batch_size']}",
        f"num_epochs: {job['num_epochs']}",
        f"imgsz: {job['imgsz']}",
        f"patience: {job['patience']}",
        f"seed: {job.get('seed', 42)}",
        f"device: {job.get('device', 'auto')}",
        f"workers: {job.get('workers', 0)}",
        "",
        "SPLIT POR CLASSE",
        "-" * 72,
    ]
    for class_name, class_counts in counts.items():
        lines.append(
            f"{class_name}: train={class_counts['train']} | "
            f"val={class_counts['val']} | test={class_counts['test']}"
        )

    lines.extend([
        "",
        "METRICAS",
        "-" * 72,
        f"top1: {top1 if top1 is not None else 'n/d'}",
        f"top5: {top5 if top5 is not None else 'n/d'}",
        f"accuracy_test: {evaluation['accuracy']:.2f}%",
        f"test_images: {evaluation['test_images']}",
        f"avg_inference_ms: {evaluation['avg_inference_ms']:.3f}",
        f"run_dir: {save_dir if save_dir else 'n/d'}",
    ])

    lines.extend([
        "",
        "METRICAS POR CLASSE",
        "-" * 72,
    ])
    for item in evaluation["classification_report"]:
        lines.append(
            f"{item['class']}: "
            f"precision={item['precision']:.2f}% | "
            f"recall={item['recall']:.2f}% | "
            f"f1={item['f1']:.2f}% | "
            f"support={item['support']}"
        )

    lines.extend([
        "",
        "MATRIZ DE CONFUSAO",
        "-" * 72,
    ])
    for row in evaluation["confusion_matrix"]:
        lines.append(" ".join(str(value) for value in row))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _run_job(job: dict[str, Any]) -> Path:
    from ultralytics import YOLO

    dataset_path, counts = _prepare_yolo_classification_dataset(job)

    print()
    print("=" * 72)
    print(
        f"{job['model_name']} | dataset={dataset_path.name} | "
        f"epochs={job['num_epochs']} | batch={job['batch_size']} | imgsz={job['imgsz']}"
    )
    print("=" * 72)

    model = YOLO(job["weights"])
    train_args = {
        "data": str(dataset_path),
        "epochs": int(job["num_epochs"]),
        "imgsz": int(job["imgsz"]),
        "batch": int(job["batch_size"]),
        "patience": int(job["patience"]),
        "seed": int(job.get("seed", 42)),
        "workers": int(job.get("workers", 0)),
        "project": str(MODELS_SAVE_DIR / "yolo_runs"),
        "name": f"soybean_{job['model_name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": False,
    }
    if job.get("device") is not None:
        train_args["device"] = job["device"]

    result = model.train(**train_args)

    save_dir = Path(getattr(result, "save_dir", ""))
    best_path = save_dir / "weights" / "best.pt"
    print("[status] Avaliando best.pt no split de teste...")
    evaluation = _evaluate_best_model(best_path, dataset_path)

    report_path = _write_report(job, dataset_path, counts, result, evaluation)
    print(f"[report] {report_path}")
    return report_path


def main() -> int:
    started_at = time.time()
    failed_jobs = 0

    for job in PIPELINE:
        try:
            _run_job(job)
        except Exception as exc:
            failed_jobs += 1
            print(f"[error] {job.get('model_name', 'YOLO')} falhou: {exc}")

    elapsed = time.time() - started_at
    print()
    print(f"Pipeline YOLO concluido em {elapsed:.1f}s. Falhas: {failed_jobs}.")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
