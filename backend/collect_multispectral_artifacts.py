from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RUNS_DIR = WORKSPACE_ROOT / "models" / "pipeline_runs"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copia modelos e relatorios dos treinamentos multispectral concluidos, "
            "removendo duplicados por modelo e seed."
        )
    )
    parser.add_argument(
        "--run-file",
        action="append",
        help=(
            "Arquivo pipeline_run_*.json a considerar. Pode repetir. "
            "Padrao: usa todos os pipeline_run_*.json com jobs multispectral."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Diretorio de saida. Padrao: "
            "models/selected_multispectral_training_<timestamp>."
        ),
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Espera ate nao haver jobs multispectral running/pending nos run files.",
    )
    parser.add_argument(
        "--wait-run-file",
        action="append",
        help=(
            "Run file usado para checar running/pending. Pode repetir. "
            "Padrao: usa apenas o pipeline_run mais recente entre os run files."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Intervalo de espera quando --wait estiver ativo. Padrao: 60.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permite copiar mesmo quando ainda ha jobs running/pending.",
    )
    return parser.parse_args()


def _resolve_run_file(path: str) -> Path:
    run_path = Path(path)
    if not run_path.is_absolute():
        run_path = WORKSPACE_ROOT / run_path
    return run_path.resolve()


def _load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_multispectral_job(job: dict[str, Any]) -> bool:
    return (
        job.get("dataset_name") == "multispectral"
        or "multispectral" in str(job.get("id", ""))
        or "multispectral" in str(job.get("data_path", ""))
    )


def _default_run_files() -> list[Path]:
    run_files: list[Path] = []
    for path in sorted(PIPELINE_RUNS_DIR.glob("pipeline_run_*.json")):
        try:
            run = _load_run(path)
        except (OSError, json.JSONDecodeError):
            continue
        if any(_is_multispectral_job(entry.get("job", {})) for entry in run.get("jobs", [])):
            run_files.append(path)
    return run_files


def _pending_multispectral_jobs(run_files: list[Path]) -> list[tuple[Path, str, str]]:
    pending: list[tuple[Path, str, str]] = []
    for path in run_files:
        run = _load_run(path)
        for entry in run.get("jobs", []):
            job = entry.get("job", {})
            status = entry.get("status")
            if _is_multispectral_job(job) and status in {"pending", "running"}:
                pending.append((path, str(status), str(job.get("id", ""))))
    return pending


def _wait_for_completion(run_files: list[Path], poll_seconds: int) -> None:
    while True:
        pending = _pending_multispectral_jobs(run_files)
        if not pending:
            return
        print(
            f"Aguardando {len(pending)} jobs multispectral "
            f"running/pending. Proxima checagem em {poll_seconds}s.",
            flush=True,
        )
        time.sleep(max(1, poll_seconds))


def _default_wait_run_files(run_files: list[Path]) -> list[Path]:
    return [max(run_files, key=lambda path: path.stat().st_mtime)]


def _entry_paths(entry: dict[str, Any]) -> tuple[Path, Path]:
    model_path = entry.get("model_path") or entry.get("result", {}).get("model_path")
    report_path = entry.get("report_path") or entry.get("result", {}).get("report_path")
    if not model_path:
        raise ValueError("Entrada success sem model_path.")
    model = Path(model_path)
    if not model.is_absolute():
        model = WORKSPACE_ROOT / model
    if report_path:
        report = Path(report_path)
        if not report.is_absolute():
            report = WORKSPACE_ROOT / report
    else:
        report = model.with_suffix(".txt")
    return model.resolve(), report.resolve()


def _dedupe_key(job: dict[str, Any]) -> tuple[str, str]:
    return str(job.get("model_name", "")), str(job.get("seed", job.get("stat_seed", "")))


def _entry_preference(row: dict[str, Any]) -> tuple[int, float, str]:
    job_id = str(row["job"].get("id", ""))
    prefer_source = 1 if "from_sem_fundo" not in job_id else 0
    macro_f1 = float(row["entry"].get("result", {}).get("macro_f1", 0.0) or 0.0)
    run_id = str(row["run"].get("run_id", ""))
    return prefer_source, macro_f1, run_id


def _collect_success_entries(run_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in run_files:
        run = _load_run(path)
        for entry in run.get("jobs", []):
            job = entry.get("job", {})
            if entry.get("status") != "success" or not _is_multispectral_job(job):
                continue
            model_path, report_path = _entry_paths(entry)
            if not model_path.is_file():
                raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
            if not report_path.is_file():
                raise FileNotFoundError(f"Relatorio nao encontrado: {report_path}")
            rows.append(
                {
                    "run_file": path,
                    "run": run,
                    "entry": entry,
                    "job": job,
                    "model_path": model_path,
                    "report_path": report_path,
                }
            )
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _dedupe_key(row["job"])
        current = chosen.get(key)
        if current is None or _entry_preference(row) > _entry_preference(current):
            chosen[key] = row
    return sorted(
        chosen.values(),
        key=lambda row: (
            str(row["job"].get("model_name", "")),
            int(row["job"].get("seed", row["job"].get("stat_seed", 0))),
        ),
    )


def _safe_filename(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in safe.split("_") if part)


def _copy_artifacts(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        job = row["job"]
        entry = row["entry"]
        result = entry.get("result", {})
        model = row["model_path"]
        report = row["report_path"]
        prefix = _safe_filename(
            f"{job.get('model_name')}_seed{job.get('seed', job.get('stat_seed'))}"
        )
        dest_model = output_dir / f"{prefix}{model.suffix}"
        dest_report = output_dir / f"{prefix}{report.suffix}"
        shutil.copy2(model, dest_model)
        shutil.copy2(report, dest_report)
        manifest_rows.append(
            {
                "job_id": job.get("id", ""),
                "model_name": job.get("model_name", ""),
                "seed": job.get("seed", job.get("stat_seed", "")),
                "dataset_name": job.get("dataset_name", ""),
                "experiment_name": job.get("experiment_name", ""),
                "status": entry.get("status", ""),
                "accuracy": result.get("accuracy", ""),
                "macro_f1": result.get("macro_f1", ""),
                "best_val_loss": result.get("best_val_loss", ""),
                "best_checkpoint_score": result.get("best_checkpoint_score", ""),
                "total_time": result.get("total_time", ""),
                "source_run_id": row["run"].get("run_id", ""),
                "source_run_file": str(row["run_file"]),
                "source_model_path": str(model),
                "source_report_path": str(report),
                "copied_model_path": str(dest_model),
                "copied_report_path": str(dest_report),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def main() -> int:
    args = _parse_args()
    run_files = (
        [_resolve_run_file(path) for path in args.run_file]
        if args.run_file
        else _default_run_files()
    )
    if not run_files:
        raise SystemExit("Nenhum pipeline_run multispectral encontrado.")
    wait_run_files = (
        [_resolve_run_file(path) for path in args.wait_run_file]
        if args.wait_run_file
        else _default_wait_run_files(run_files)
    )
    if args.wait:
        _wait_for_completion(wait_run_files, args.poll_seconds)
    pending = _pending_multispectral_jobs(wait_run_files)
    if pending and not args.allow_incomplete:
        print("Ainda ha jobs multispectral running/pending:")
        for path, status, job_id in pending:
            print(f"- {path.name}: {status} {job_id}")
        raise SystemExit(
            "Use --wait para aguardar ou --allow-incomplete para copiar apenas success."
        )
    rows = _collect_success_entries(run_files)
    if not rows:
        raise SystemExit("Nenhum job multispectral success encontrado.")
    output_dir = (
        _resolve_run_file(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
        / f"selected_multispectral_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    manifest_path = _copy_artifacts(rows, output_dir)
    print(f"Copiados {len(rows)} modelos e {len(rows)} relatorios para {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
