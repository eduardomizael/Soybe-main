from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = WORKSPACE_ROOT / "models"
RUNS_DIR = MODELS_DIR / "pipeline_runs"
RGB_DIR = MODELS_DIR / "rgb_seed_expansion_10_analysis_with_convnext"
ABLATION_DIR = MODELS_DIR / "mobilenetv3_ablation_3seeds_analysis"
DEFAULT_OUTPUT = MODELS_DIR / "article1_rgb_enxuto_package_20260723"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empacota os artefatos finais do Artigo 1 RGB enxuto.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Artefato ausente: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Cópia inválida: {destination}")


def _representatives(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for factor in sorted({row["fator"] for row in rows}):
        factor_rows = [row for row in rows if row["fator"] == factor]
        target = median(float(row["macro_f1"]) for row in factor_rows)
        chosen = min(
            factor_rows,
            key=lambda row: (abs(float(row["macro_f1"]) - target), int(row["semente"])),
        )
        selected.append({**chosen, "criterio": "macro_f1_mais_proximo_da_mediana_do_fator"})
    return selected


def _final_report(rgb_summary: list[dict[str, str]], ablation_summary: list[dict[str, str]]) -> str:
    by_model = {row["arquitetura"]: row for row in rgb_summary}
    convnext = by_model["ConvNeXtTiny"]
    resnet = by_model["ResNet50"]
    scheduler = next(row for row in ablation_summary if row["fator"] == "scheduler_name")
    return "\n".join(
        [
            "# Pacote final — Artigo 1 RGB (versão enxuta)",
            "",
            "## Escopo concluído",
            "",
            "- Comparação RGB `com_fundo`: seis arquiteturas, dez seeds, 60 execuções.",
            "- ConvNeXt-Tiny: dez seeds, incluída no mesmo Friedman e Wilcoxon-Holm.",
            "- Ablação MobileNetV3: controle e sete fatores em três seeds, 24 combinações.",
            "",
            "## Resultados-chave",
            "",
            f"- ConvNeXt-Tiny: acurácia média {float(convnext['accuracy_media']):.2f}% e macro-F1 {float(convnext['macro_f1_media']):.2f}%.",
            f"- ResNet50: acurácia média {float(resnet['accuracy_media']):.2f}% e macro-F1 {float(resnet['macro_f1_media']):.2f}%.",
            f"- Scheduler ReduceLROnPlateau: delta médio de {float(scheduler['delta_accuracy_medio_pp']):+.2f} p.p. em acurácia e {float(scheduler['delta_macro_f1_medio_pp']):+.2f} p.p. em macro-F1 contra o controle.",
            "- A ablação tem somente três pares por fator; seus resultados devem ser descritos como consistência observada, não como significância estatística forte.",
            "",
            "## Conteúdo do pacote",
            "",
            "- `models/` e `reports/`: checkpoints e relatórios individuais copiados, com origem registrada no manifesto.",
            "- `analysis/rgb/`: consolidação de seis arquiteturas e dez seeds.",
            "- `analysis/ablation/`: dados, deltas pareados e testes da ablação.",
            "- `article_csv/`: tabelas no esquema solicitado para integração ao artigo.",
            "- `source/`: filas, scripts e run-files usados nas execuções.",
            "",
        ]
    )


def main() -> int:
    args = _parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Diretório de saída já existe: {output}")

    rgb_rows = _read_csv(RGB_DIR / "rgb_seed_runs.csv")
    ablation_rows = _read_csv(ABLATION_DIR / "ablation_runs.csv")
    rgb_summary = _read_csv(RGB_DIR / "rgb_seed_summary.csv")
    ablation_summary = _read_csv(ABLATION_DIR / "ablation_factor_summary.csv")
    if len(rgb_rows) != 60 or len(ablation_rows) != 24:
        raise ValueError(f"Contagens inesperadas: RGB={len(rgb_rows)}, ablação={len(ablation_rows)}.")

    model_dir = output / "models"
    report_dir = output / "reports"
    analysis_dir = output / "analysis"
    article_csv_dir = output / "article_csv"
    source_dir = output / "source"
    for directory in (model_dir, report_dir, analysis_dir, article_csv_dir, source_dir):
        directory.mkdir(parents=True)

    manifest: list[dict[str, str]] = []
    for group, rows in (("rgb_6arquiteturas_10seeds", rgb_rows), ("ablacao_mobilenetv3_3seeds", ablation_rows)):
        for row in rows:
            model_source = Path(row["model_path"])
            report_source = Path(row["report_path"])
            prefix = f"{group}__"
            model_dest = model_dir / f"{prefix}{model_source.name}"
            report_dest = report_dir / f"{prefix}{report_source.name}"
            _copy(model_source, model_dest)
            _copy(report_source, report_dest)
            manifest.append(
                {
                    "grupo": group,
                    "job_id": row.get("job_id", ""),
                    "arquitetura": row.get("arquitetura", "MobileNetV3"),
                    "conjunto": row.get("conjunto", "com_fundo"),
                    "semente": row["semente"],
                    "acuracia": row["acuracia"],
                    "macro_f1": row["macro_f1"],
                    "fator": row.get("fator", ""),
                    "nivel": row.get("nivel", ""),
                    "source_model_path": str(model_source),
                    "copied_model_path": str(model_dest.relative_to(output)),
                    "source_report_path": str(report_source),
                    "copied_report_path": str(report_dest.relative_to(output)),
                }
            )

    for name in ("rgb_seed_runs.csv", "rgb_seed_summary.csv", "rgb_seed_statistical_tests.csv", "rgb_seed_representative_runs.csv", "analysis_metadata.json"):
        _copy(RGB_DIR / name, analysis_dir / "rgb" / name)
    for name in ("ablation_runs.csv", "ablation_factor_summary.csv", "ablation_paired_tests.csv", "analysis_metadata.json", "relatorio_consolidado_ablation_3seeds.md"):
        _copy(ABLATION_DIR / name, analysis_dir / "ablation" / name)

    rgb_article = [
        {
            "experimento": "rgb_6_arquiteturas_10_seeds",
            "arquitetura": row["arquitetura"],
            "conjunto": row["conjunto"],
            "semente": row["semente"],
            "acuracia": row["acuracia"],
            "macro_f1": row["macro_f1"],
        }
        for row in rgb_rows
    ]
    ablation_article = [
        {
            "experimento": f"ablacao_{row['fator']}",
            "arquitetura": "MobileNetV3",
            "conjunto": "com_fundo",
            "semente": row["semente"],
            "acuracia": row["acuracia"],
            "macro_f1": row["macro_f1"],
        }
        for row in ablation_rows
    ]
    article_fields = ["experimento", "arquitetura", "conjunto", "semente", "acuracia", "macro_f1"]
    _write_csv(article_csv_dir / "rgb_6_arquiteturas_10_seeds.csv", rgb_article, article_fields)
    _write_csv(article_csv_dir / "ablacao_7_fatores_3_seeds.csv", ablation_article, article_fields)
    _write_csv(
        article_csv_dir / "ablacao_representative_runs.csv",
        _representatives(ablation_rows),
        [*ablation_rows[0].keys(), "criterio"],
    )

    source_files = (
        WORKSPACE_ROOT / "backend" / "training_jobs_convnext_tiny_com_fundo_10_seeds.toml",
        WORKSPACE_ROOT / "backend" / "training_jobs_mobilenetv3_com_fundo_ablation_completion.toml",
        WORKSPACE_ROOT / "backend" / "analyze_rgb_seed_expansion.py",
        WORKSPACE_ROOT / "backend" / "analyze_mobilenet_ablation.py",
        WORKSPACE_ROOT / "backend" / "generate_convnext_tiny_rgb_jobs.py",
        WORKSPACE_ROOT / "backend" / "generate_mobilenet_ablation_completion_jobs.py",
        WORKSPACE_ROOT / "scripts" / "run_convnext_tiny_rgb_10_seeds.ps1",
        WORKSPACE_ROOT / "scripts" / "run_mobilenet_ablation_completion.ps1",
        RUNS_DIR / "pipeline_run_20260721_181721.json",
        RUNS_DIR / "pipeline_run_20260722_180341.json",
    )
    for source in source_files:
        _copy(source, source_dir / source.name)

    _write_csv(
        output / "manifest.csv",
        manifest,
        list(manifest[0]),
    )
    (output / "relatorio_final_artigo1_rgb_enxuto.md").write_text(
        _final_report(rgb_summary, ablation_summary), encoding="utf-8"
    )
    metadata = {
        "rgb_runs": len(rgb_rows),
        "ablation_runs": len(ablation_rows),
        "models": len(list(model_dir.glob("*.pth"))),
        "reports": len(list(report_dir.glob("*.txt"))),
        "manifest_rows": len(manifest),
    }
    (output / "package_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if metadata["models"] != 84 or metadata["reports"] != 84 or metadata["manifest_rows"] != 84:
        raise RuntimeError(f"Validação de pacote falhou: {metadata}")
    print(f"Pacote criado em {output}: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
