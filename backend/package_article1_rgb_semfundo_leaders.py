from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ANALYSIS = MODELS / "rgb_background_effect_leaders_10seeds"
DEFAULT_OUTPUT = MODELS / "article1_rgb_semfundo_leaders_package_20260723"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empacota a comparação RGB com_fundo versus sem_fundo.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file(): raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size == 0: raise RuntimeError(f"Arquivo vazio: {destination}")


def _representatives(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for model in sorted({r["arquitetura"] for r in rows}):
        for dataset in ("com_fundo", "sem_fundo"):
            group = [r for r in rows if r["arquitetura"] == model and r["conjunto"] == dataset]
            center = median(float(r["macro_f1"]) for r in group)
            chosen = min(group, key=lambda r: (abs(float(r["macro_f1"]) - center), int(r["semente"])))
            selected.append({**chosen, "criterio": "macro_f1_mais_proximo_da_mediana"})
    return selected


def main() -> int:
    args = _args(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f"Diretório já existe: {output}")
    rows = _read(ANALYSIS / "background_effect_runs.csv")
    summary = _read(ANALYSIS / "background_effect_summary.csv")
    if len(rows) != 40: raise ValueError(f"Esperadas 40 execuções; encontradas {len(rows)}")
    for part in ("models", "reports", "analysis", "article_csv", "source"):
        (output / part).mkdir(parents=True)
    manifest = []
    for row in rows:
        model_source, report_source = Path(row["model_path"]), Path(row["report_path"])
        prefix = f"{row['conjunto']}__"
        model_dest, report_dest = output / "models" / f"{prefix}{model_source.name}", output / "reports" / f"{prefix}{report_source.name}"
        _copy(model_source, model_dest); _copy(report_source, report_dest)
        manifest.append({
            "experimento": row["experimento"], "arquitetura": row["arquitetura"], "conjunto": row["conjunto"], "semente": row["semente"],
            "acuracia": row["acuracia"], "macro_f1": row["macro_f1"], "job_id": row["job_id"],
            "source_model_path": str(model_source), "copied_model_path": str(model_dest.relative_to(output)),
            "source_report_path": str(report_source), "copied_report_path": str(report_dest.relative_to(output)),
        })
    for item in ANALYSIS.iterdir():
        if item.is_file(): _copy(item, output / "analysis" / item.name)
    _write(output / "article_csv" / "rgb_background_effect_runs.csv", rows, ["experimento", "arquitetura", "conjunto", "semente", "acuracia", "macro_f1", "job_id", "finished_at", "run_file", "model_path", "report_path"])
    _write(output / "article_csv" / "representative_runs.csv", _representatives(rows))
    _write(output / "manifest.csv", manifest)
    for file in (
        ROOT / "backend" / "training_jobs_rgb_semfundo_leaders_remaining_10.toml",
        ROOT / "backend" / "generate_rgb_semfundo_leaders_remaining_jobs.py",
        ROOT / "backend" / "analyze_rgb_background_effect.py",
        ROOT / "scripts" / "run_rgb_semfundo_leaders_remaining.ps1",
        MODELS / "pipeline_runs" / "pipeline_run_20260723_092203.json",
        MODELS / "pipeline_runs" / "pipeline_run_20260715_092900.json",
        MODELS / "pipeline_runs" / "pipeline_run_20260606_162821.json",
    ): _copy(file, output / "source" / file.name)
    result = {r["arquitetura"]: r for r in summary}
    report = [
        "# Comparação RGB — efeito da remoção de fundo", "",
        "## Integridade", "", "- 40/40 execuções disponíveis: duas arquiteturas, dois conjuntos e dez seeds.",
        "- 20/20 pares por seed disponíveis.", "", "## Resultados", "",
        "| Arquitetura | Delta acurácia sem-com (p.p.) | IC95% | Delta macro-F1 sem-com (p.p.) | IC95% | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in result.items():
        report.append(f"| {model} | {float(row['delta_acuracia_medio_pp']):+.2f} | [{float(row['ic95_acuracia_inferior_pp']):+.2f}, {float(row['ic95_acuracia_superior_pp']):+.2f}] | {float(row['delta_macro_f1_medio_pp']):+.2f} | [{float(row['ic95_macro_f1_inferior_pp']):+.2f}, {float(row['ic95_macro_f1_superior_pp']):+.2f}] | {float(row['wilcoxon_macro_f1_p']):.5f} |")
    report.extend(["", "## Interpretação", "", "A remoção de fundo reduziu acurácia e macro-F1 em todas as dez repetições de ambas as arquiteturas. O pareamento é por seed/replicação; os conjuntos possuem números distintos de imagens, portanto não se afirma pareamento imagem a imagem.", ""])
    (output / "relatorio_final_background_effect.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {"runs": len(rows), "models": len(list((output / 'models').glob('*.pth'))), "reports": len(list((output / 'reports').glob('*.txt'))), "manifest_rows": len(manifest)}
    (output / "package_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if list(metadata.values()) != [40, 40, 40, 40]: raise RuntimeError(metadata)
    print(f"Pacote criado em {output}: {metadata}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
