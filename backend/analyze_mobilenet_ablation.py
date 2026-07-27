from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKSPACE_ROOT / "models" / "pipeline_runs"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "models" / "mobilenetv3_ablation_3seeds_analysis"
SEEDS = (42, 1337, 2026)
FACTORS = (
    "split_strategy",
    "sampler_strategy",
    "optimizer_name",
    "scheduler_name",
    "freeze_backbone_epochs",
    "checkpoint_metric",
    "num_epochs",
)
METRICS = ("accuracy", "macro_f1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida a ablação MobileNetV3 em três seeds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for run_file in sorted(RUNS_DIR.glob("pipeline_run_*.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        for entry in run.get("jobs", []):
            job = entry.get("job", {})
            result = entry.get("result", {}) or {}
            if (
                entry.get("status") != "success"
                or job.get("model_name") != "MobileNetV3"
                or job.get("dataset_name") != "com_fundo"
                or job.get("ablation_role") not in {"control", "factor"}
                or int(job.get("seed", -1)) not in SEEDS
                or not all(metric in result for metric in METRICS)
            ):
                continue
            factor = str(job.get("ablation_factor", "none"))
            key = factor, int(job["seed"])
            row = {
                "fator": factor,
                "nivel": job.get("ablation_level", "baseline"),
                "semente": int(job["seed"]),
                "acuracia": float(result["accuracy"]),
                "macro_f1": float(result["macro_f1"]),
                "job_id": job.get("id", ""),
                "finished_at": entry.get("finished_at", ""),
                "run_file": str(run_file.resolve()),
                "report_path": entry.get("report_path", ""),
                "model_path": entry.get("model_path", ""),
            }
            if key not in selected or row["finished_at"] > selected[key]["finished_at"]:
                selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["fator"], row["semente"]))


def _summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["fator"], row["semente"]): row for row in rows}
    summary: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for factor in FACTORS:
        pairs = [
            (lookup[("none", seed)], lookup[(factor, seed)])
            for seed in SEEDS
            if ("none", seed) in lookup and (factor, seed) in lookup
        ]
        item: dict[str, Any] = {
            "fator": factor,
            "pares_concluidos": len(pairs),
            "pares_esperados": len(SEEDS),
            "seeds_concluidas": ",".join(str(factor_row["semente"]) for _, factor_row in pairs),
        }
        for metric, label in (("acuracia", "accuracy"), ("macro_f1", "macro_f1")):
            deltas = np.array([factor_row[metric] - control[metric] for control, factor_row in pairs])
            item[f"delta_{label}_medio_pp"] = float(deltas.mean()) if len(deltas) else ""
            item[f"delta_{label}_dp_pp"] = float(deltas.std(ddof=1)) if len(deltas) > 1 else ""
            if len(deltas) >= 2:
                statistic, p_value = (0.0, 1.0) if np.allclose(deltas, 0) else wilcoxon(deltas)
                tests.append({
                    "fator": factor,
                    "metrica": label,
                    "n_pares": len(deltas),
                    "estatistica": float(statistic),
                    "p_valor": float(p_value),
                })
        summary.append(item)
    return summary, tests


def _report(summary: list[dict[str, Any]], missing: list[str]) -> str:
    lines = [
        "# Consolidação parcial — ablação MobileNetV3 RGB em três seeds",
        "",
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Cada fator é comparado pareadamente ao controle da mesma seed. Os deltas são em pontos percentuais.",
        "",
        "| Fator | Pares concluídos | Delta acurácia médio | Delta macro-F1 médio |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['fator']} | {row['pares_concluidos']}/{row['pares_esperados']} | "
            f"{row['delta_accuracy_medio_pp'] if row['delta_accuracy_medio_pp'] != '' else '—'} | "
            f"{row['delta_macro_f1_medio_pp'] if row['delta_macro_f1_medio_pp'] != '' else '—'} |"
        )
    lines.extend(["", "## Integridade", ""])
    if missing:
        lines.append(f"Ainda faltam {len(missing)} combinações fator-seed: {', '.join(missing)}.")
        lines.append("Não interpretar testes inferenciais antes de completar os três pares por fator.")
    else:
        lines.append("As 24 combinações esperadas (controle + sete fatores × três seeds) foram encontradas.")
        lines.append("Com três pares, os testes devem ser relatados como evidência de consistência, não como prova forte de significância.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    found = {(row["fator"], row["semente"]) for row in rows}
    expected = {(factor, seed) for factor in ("none", *FACTORS) for seed in SEEDS}
    missing = [f"{factor}:seed{seed}" for factor, seed in sorted(expected - found)]
    summary, tests = _summary(rows)
    _write_csv(args.output_dir / "ablation_runs.csv", rows)
    _write_csv(args.output_dir / "ablation_factor_summary.csv", summary)
    _write_csv(args.output_dir / "ablation_paired_tests.csv", tests)
    (args.output_dir / "relatorio_consolidado_ablation_3seeds.md").write_text(
        _report(summary, missing), encoding="utf-8"
    )
    metadata = {"expected_runs": len(expected), "found_runs": len(found), "missing": missing}
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Consolidação salva em {args.output_dir}: {len(found)}/{len(expected)} execuções.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
