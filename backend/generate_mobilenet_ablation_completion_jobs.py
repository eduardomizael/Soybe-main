from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_mobilenetv3_com_fundo_ablation.toml"
OUTPUT_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_mobilenetv3_com_fundo_ablation_completion.toml"
TARGET_SEEDS = (1337, 2026)
EXCLUDED_FACTOR = "scheduler_name"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _render(jobs: list[dict[str, Any]]) -> str:
    lines = [
        "# Completa a ablação MobileNetV3 RGB em três seeds.",
        "# Os controles e o fator scheduler_name das seeds 1337/2026 já foram",
        "# concluídos em training_jobs_mobilenetv3_com_fundo_ablation_confirmation.toml.",
        "# Esta fila contém somente os seis fatores restantes x duas seeds = 12 jobs.",
        "",
    ]
    for job in jobs:
        lines.append("[[jobs]]")
        for key, value in job.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    with SOURCE_PATH.open("rb") as file:
        source_jobs = tomllib.load(file)["jobs"]

    factor_jobs = [
        job for job in source_jobs
        if job.get("ablation_role") == "factor"
        and job.get("ablation_factor") != EXCLUDED_FACTOR
    ]
    if len(factor_jobs) != 6:
        raise ValueError(f"Esperados seis fatores fora scheduler_name; encontrados {len(factor_jobs)}.")

    jobs: list[dict[str, Any]] = []
    for seed in TARGET_SEEDS:
        for source in factor_jobs:
            job = deepcopy(source)
            job["id"] = job["id"].replace("seed42", f"seed{seed}")
            job["experiment_name"] = job["experiment_name"].replace("seed42", f"seed{seed}")
            job["reference_job_id"] = f"mobilenetv3_com_fundo_ablation_control_seed{seed}"
            job["seed"] = seed
            job["tags"] = [
                f"seed-{seed}" if tag == "seed-42" else tag
                for tag in job["tags"]
            ] + ["ablation-completion"]
            job["notes"] = (
                f"Confirmacao da ablação do fator {job['ablation_factor']} na seed {seed}. "
                "Mantém a configuração da rodada exploratória e altera somente esse fator."
            )
            jobs.append(job)

    OUTPUT_PATH.write_text(_render(jobs), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
