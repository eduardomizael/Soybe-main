from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_estatistic_analisis.toml"
OUTPUT_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_convnext_tiny_com_fundo_10_seeds.toml"
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def main() -> int:
    with SOURCE_PATH.open("rb") as file:
        source_jobs = tomllib.load(file)["jobs"]
    source = next(
        job for job in source_jobs
        if job.get("model_name") == "ResNet50"
        and job.get("dataset_name") == "com_fundo"
        and job.get("seed") == 42
        and "best_candidate" in str(job.get("experiment_name", ""))
    )

    jobs: list[dict[str, Any]] = []
    for repeat, seed in enumerate(SEEDS, start=1):
        job = deepcopy(source)
        group = "convnexttiny_com_fundo_best_candidate"
        job.update(
            {
                "id": f"{group}_seed{seed}",
                "tags": [
                    "best-candidate", "com_fundo", "convnext", "convnext-tiny",
                    "macro-f1", "stratified", "weighted", "statistical-repeat",
                    "rgb-convnext-tiny-10-seeds", f"seed-{seed}", f"repeat-{repeat:02d}",
                ],
                "notes": (
                    "ConvNeXt-Tiny RGB com_fundo, mesma receita experimental da ResNet50; "
                    f"seed {seed}, repetição {repeat}/10."
                ),
                "stat_group": group,
                "stat_repeat": repeat,
                "stat_seed": seed,
                "stat_total_repeats": 10,
                "source_job_id": "resnet50_com_fundo_best_candidate",
                "model_name": "ConvNeXtTiny",
                "experiment_name": f"best_candidate_com_fundo_seed{seed}",
                "seed": seed,
            }
        )
        jobs.append(job)

    lines = [
        "# ConvNeXt-Tiny no conjunto RGB com_fundo: 10 seeds canônicas.",
        "# Receita base: ResNet50 best_candidate; modelo substituído por ConvNeXtTiny.",
        "",
    ]
    for job in jobs:
        lines.append("[[jobs]]")
        for key, value in job.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
