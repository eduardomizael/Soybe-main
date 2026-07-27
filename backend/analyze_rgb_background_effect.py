from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "pipeline_runs"
OUTPUT = ROOT / "models" / "rgb_background_effect_leaders_10seeds"
MODELS = ("ResNet50", "MobileNetV3")
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara com_fundo e sem_fundo por seed.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _runs() -> list[dict[str, Any]]:
    chosen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in RUNS.glob("pipeline_run_*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        for entry in run.get("jobs", []):
            job, result = entry.get("job", {}), entry.get("result", {}) or {}
            key = (job.get("model_name"), job.get("dataset_name"), int(job.get("seed", -1)))
            if entry.get("status") != "success" or key[0] not in MODELS or key[1] not in {"com_fundo", "sem_fundo"} or key[2] not in SEEDS:
                continue
            if "accuracy" not in result or "macro_f1" not in result or "best_candidate" not in str(job.get("experiment_name", "")):
                continue
            row = {
                "experimento": "background_effect_leaders_10_seeds",
                "arquitetura": key[0], "conjunto": key[1], "semente": key[2],
                "acuracia": float(result["accuracy"]), "macro_f1": float(result["macro_f1"]),
                "job_id": job.get("id", ""), "finished_at": entry.get("finished_at", ""),
                "run_file": str(path.resolve()), "model_path": entry.get("model_path", ""), "report_path": entry.get("report_path", ""),
            }
            old = chosen.get(key)
            if old is None or (row["finished_at"], row["run_file"]) > (old["finished_at"], old["run_file"]): chosen[key] = row
    return sorted(chosen.values(), key=lambda r: (MODELS.index(r["arquitetura"]), r["conjunto"], SEEDS.index(r["semente"])))


def _bootstrap(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(20260723)
    means = rng.choice(np.asarray(values), size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main() -> int:
    args = _args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _runs(); lookup = {(r["arquitetura"], r["conjunto"], r["semente"]): r for r in rows}
    deltas: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            com, sem = lookup.get((model, "com_fundo", seed)), lookup.get((model, "sem_fundo", seed))
            if com and sem:
                deltas.append({
                    "arquitetura": model, "semente": seed,
                    "acuracia_com_fundo": com["acuracia"], "acuracia_sem_fundo": sem["acuracia"],
                    "delta_acuracia_sem_menos_com_pp": sem["acuracia"] - com["acuracia"],
                    "macro_f1_com_fundo": com["macro_f1"], "macro_f1_sem_fundo": sem["macro_f1"],
                    "delta_macro_f1_sem_menos_com_pp": sem["macro_f1"] - com["macro_f1"],
                })
    summary: list[dict[str, Any]] = []
    for model in MODELS:
        model_deltas = [r for r in deltas if r["arquitetura"] == model]
        item: dict[str, Any] = {"arquitetura": model, "pares_concluidos": len(model_deltas), "pares_esperados": len(SEEDS)}
        for metric in ("acuracia", "macro_f1"):
            values = [r[f"delta_{metric}_sem_menos_com_pp"] for r in model_deltas]
            item[f"delta_{metric}_medio_pp"] = float(np.mean(values)) if values else ""
            if values: item[f"ic95_{metric}_inferior_pp"], item[f"ic95_{metric}_superior_pp"] = _bootstrap(values)
            else: item[f"ic95_{metric}_inferior_pp"], item[f"ic95_{metric}_superior_pp"] = "", ""
            if len(values) >= 2:
                stat, p = (0.0, 1.0) if np.allclose(values, 0) else wilcoxon(values)
                item[f"wilcoxon_{metric}_p"] = float(p); item[f"wilcoxon_{metric}_estatistica"] = float(stat)
            else: item[f"wilcoxon_{metric}_p"], item[f"wilcoxon_{metric}_estatistica"] = "", ""
        summary.append(item)
    expected = {(m, d, s) for m in MODELS for d in ("com_fundo", "sem_fundo") for s in SEEDS}
    missing = [
        {"arquitetura": m, "conjunto": d, "semente": s}
        for m, d, s in sorted(expected - set(lookup))
    ]
    _csv(args.output_dir / "background_effect_runs.csv", rows)
    _csv(args.output_dir / "background_effect_paired_deltas.csv", deltas)
    _csv(args.output_dir / "background_effect_summary.csv", summary)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({"expected_runs": 40, "found_runs": len(rows), "paired_runs": len(deltas), "missing": missing, "pairing_note": "Pareamento por seed/replicação; os datasets têm contagens distintas e não se afirma identidade de imagens de teste."}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# Comparação com_fundo × sem_fundo — líderes RGB", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "", "O pareamento é por seed/replicação. Como os conjuntos têm números distintos de imagens, não interpretar como pareamento imagem a imagem.", "", "| Arquitetura | Pares | Delta acurácia (sem-com) | Delta macro-F1 (sem-com) |", "|---|---:|---:|---:|"]
    for item in summary: report.append(f"| {item['arquitetura']} | {item['pares_concluidos']}/10 | {item['delta_acuracia_medio_pp']} | {item['delta_macro_f1_medio_pp']} |")
    (args.output_dir / "relatorio_background_effect.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Análise salva em {args.output_dir}: {len(rows)}/40 execuções, {len(deltas)}/20 pares.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
