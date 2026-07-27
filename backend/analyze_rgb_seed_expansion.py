from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKSPACE_ROOT / "models" / "pipeline_runs"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "models" / "rgb_seed_expansion_10_analysis"
MODELS = ["ResNet50", "MobileNetV3", "EfficientNetB3", "EfficientNetB0", "EfficientNetB2"]
CONVNEXT_MODEL = "ConvNeXtTiny"
SEEDS = [42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555]
METRICS = ["accuracy", "macro_f1"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida as 10 seeds RGB e calcula IC95%, Friedman e Wilcoxon-Holm."
    )
    parser.add_argument("--run-file", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--include-convnext-tiny",
        action="store_true",
        help="Inclui ConvNeXt-Tiny como sexta arquitetura na consolidação RGB.",
    )
    return parser.parse_args()


def _model_slug(model_name: str) -> str:
    return "".join(char.lower() for char in model_name if char.isalnum())


def _is_target_job(job: dict[str, Any], models: list[str]) -> bool:
    model = str(job.get("model_name", ""))
    expected_group = f"{_model_slug(model)}_com_fundo_best_candidate"
    return (
        model in models
        and job.get("dataset_name") == "com_fundo"
        and int(job.get("seed", -1)) in SEEDS
        and job.get("stat_group") == expected_group
    )


def _resolve_path(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return str(path.resolve())


def _candidate_rows(run_files: list[Path], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_file in run_files:
        run = json.loads(run_file.read_text(encoding="utf-8"))
        for entry in run.get("jobs", []):
            job = entry.get("job", {})
            result = entry.get("result", {}) or {}
            if entry.get("status") != "success" or not _is_target_job(job, models):
                continue
            if not all(metric in result for metric in METRICS):
                continue
            rows.append(
                {
                    "experimento": "rgb_seed_expansion_10",
                    "arquitetura": job["model_name"],
                    "conjunto": job["dataset_name"],
                    "semente": int(job["seed"]),
                    "acuracia": float(result["accuracy"]),
                    "macro_f1": float(result["macro_f1"]),
                    "job_id": job.get("id", ""),
                    "finished_at": entry.get("finished_at", ""),
                    "run_id": run.get("run_id", ""),
                    "run_file": str(run_file.resolve()),
                    "report_path": _resolve_path(
                        entry.get("report_path") or result.get("report_path")
                    ),
                    "model_path": _resolve_path(
                        entry.get("model_path") or result.get("model_path")
                    ),
                }
            )
    return rows


def _dedupe(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = row["arquitetura"], row["semente"]
        current = chosen.get(key)
        preference = row["finished_at"], row["run_id"]
        if current is None or preference > (current["finished_at"], current["run_id"]):
            chosen[key] = row
    model_rank = {model: index for index, model in enumerate(models)}
    seed_rank = {seed: index for index, seed in enumerate(SEEDS)}
    return sorted(
        chosen.values(),
        key=lambda row: (model_rank[row["arquitetura"]], seed_rank[row["semente"]]),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _summary_rows(rows: list[dict[str, Any]], samples: int, models: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        model_rows = [row for row in rows if row["arquitetura"] == model]
        item: dict[str, Any] = {"arquitetura": model, "n": len(model_rows)}
        for metric_index, metric in enumerate(METRICS):
            source_key = "acuracia" if metric == "accuracy" else "macro_f1"
            values = np.array([row[source_key] for row in model_rows], dtype=float)
            low, high = _bootstrap_ci(values, samples, 20260714 + model_index * 10 + metric_index)
            item.update(
                {
                    f"{metric}_media": float(values.mean()),
                    f"{metric}_desvio_padrao": float(values.std(ddof=1)),
                    f"{metric}_mediana": float(np.median(values)),
                    f"{metric}_ic95_inferior": low,
                    f"{metric}_ic95_superior": high,
                }
            )
        summary.append(item)
    return summary


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def _paired_values(rows: list[dict[str, Any]], metric_key: str, models: list[str]) -> dict[str, np.ndarray]:
    lookup = {
        (row["arquitetura"], row["semente"]): float(row[metric_key])
        for row in rows
    }
    return {
        model: np.array([lookup[(model, seed)] for seed in SEEDS], dtype=float)
        for model in models
    }


def _test_rows(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, metric_key in [("accuracy", "acuracia"), ("macro_f1", "macro_f1")]:
        values = _paired_values(rows, metric_key, models)
        friedman = friedmanchisquare(*(values[model] for model in models))
        output.append(
            {
                "teste": "friedman",
                "metrica": metric,
                "arquitetura_a": "todas",
                "arquitetura_b": "todas",
                "n_pares": len(SEEDS),
                "estatistica": float(friedman.statistic),
                "p_valor": float(friedman.pvalue),
                "p_holm": "",
            }
        )
        pair_rows: list[dict[str, Any]] = []
        p_values: list[float] = []
        for model_a, model_b in combinations(models, 2):
            differences = values[model_a] - values[model_b]
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(values[model_a], values[model_b], alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            pair_rows.append(
                {
                    "teste": "wilcoxon",
                    "metrica": metric,
                    "arquitetura_a": model_a,
                    "arquitetura_b": model_b,
                    "n_pares": len(SEEDS),
                    "estatistica": statistic,
                    "p_valor": p_value,
                    "p_holm": 0.0,
                }
            )
            p_values.append(p_value)
        for row, adjusted in zip(pair_rows, _holm_adjust(p_values), strict=True):
            row["p_holm"] = adjusted
        output.extend(pair_rows)
    return output


def _representative_rows(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    representatives: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row["arquitetura"] == model]
        median = float(np.median([row["macro_f1"] for row in model_rows]))
        chosen = min(
            model_rows,
            key=lambda row: (abs(row["macro_f1"] - median), row["semente"]),
        )
        representatives.append({**chosen, "criterio": "macro_f1_mais_proximo_da_mediana"})
    return representatives


def main() -> int:
    args = _parse_args()
    models = [*MODELS, CONVNEXT_MODEL] if args.include_convnext_tiny else MODELS
    run_files = (
        [path.resolve() for path in args.run_file]
        if args.run_file
        else sorted(RUNS_DIR.glob("pipeline_run_*.json"))
    )
    rows = _dedupe(_candidate_rows(run_files, models), models)
    expected = {(model, seed) for model in models for seed in SEEDS}
    found = {(row["arquitetura"], row["semente"]) for row in rows}
    missing = sorted(expected - found)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rgb_seed_runs.csv", rows)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "seeds": SEEDS,
        "expected_runs": len(expected),
        "found_runs": len(found),
        "missing": [{"arquitetura": model, "semente": seed} for model, seed in missing],
        "bootstrap_samples": args.bootstrap_samples,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Analise incompleta: {len(missing)} de {len(expected)} execucoes ausentes. "
            f"Consulte {args.output_dir / 'analysis_metadata.json'}."
        )
    if missing:
        print(f"Consolidacao parcial salva: {len(found)}/{len(expected)} execucoes.")
        return 0

    _write_csv(args.output_dir / "rgb_seed_summary.csv", _summary_rows(rows, args.bootstrap_samples, models))
    _write_csv(args.output_dir / "rgb_seed_statistical_tests.csv", _test_rows(rows, models))
    _write_csv(args.output_dir / "rgb_seed_representative_runs.csv", _representative_rows(rows, models))
    print(f"Analise completa salva em {args.output_dir}: {len(rows)} execucoes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
