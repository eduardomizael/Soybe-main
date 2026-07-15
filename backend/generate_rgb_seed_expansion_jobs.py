from __future__ import annotations

import argparse
import re
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORKSPACE_ROOT / "backend" / "training_jobs_estatistic_analisis.toml"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "backend" / "training_jobs_rgb_seed_expansion_10.toml"
EXISTING_SEEDS = [42, 1337, 2026, 9001]
NEW_SEEDS = [7, 123, 2024, 31337, 777, 555]
MODEL_ORDER = ["ResNet50", "MobileNetV3", "EfficientNetB3", "EfficientNetB0", "EfficientNetB2"]
LEADER_MODELS = {"ResNet50", "MobileNetV3", "EfficientNetB3"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera os 30 jobs RGB que completam 10 seeds.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _slug_model(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_name.lower())


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Valor TOML nao suportado: {value!r}")


def _source_jobs(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as file:
        jobs = tomllib.load(file).get("jobs", [])
    selected = {
        job["model_name"]: job
        for job in jobs
        if job.get("dataset_name") == "com_fundo"
        and job.get("seed") == 42
        and job.get("model_name") in MODEL_ORDER
    }
    missing = [model for model in MODEL_ORDER if model not in selected]
    if missing:
        raise ValueError(f"Jobs fonte ausentes: {', '.join(missing)}")
    return selected


def _expanded_job(source: dict[str, Any], model_name: str, seed: int, repeat: int) -> dict[str, Any]:
    job = deepcopy(source)
    model_slug = _slug_model(model_name)
    group = f"{model_slug}_com_fundo_best_candidate"
    job["id"] = f"{group}_seed{seed}"
    tags = [
        tag
        for tag in job.get("tags", [])
        if not str(tag).startswith("seed-") and not str(tag).startswith("repeat-")
    ]
    tags.extend(
        [
            "rgb-seed-expansion-10",
            "leaders-first" if model_name in LEADER_MODELS else "remaining-models",
            f"seed-{seed}",
            f"repeat-{repeat:02d}",
        ]
    )
    job["tags"] = list(dict.fromkeys(tags))
    job["notes"] = (
        f"Expansao RGB para 10 seeds, repeticao {repeat}/10. "
        f"Replica a receita experimental validada de {model_name} em com_fundo; "
        f"somente a seed muda para {seed}."
    )
    job["stat_group"] = group
    job["stat_repeat"] = repeat
    job["stat_seed"] = seed
    job["stat_total_repeats"] = 10
    job["source_job_id"] = group
    job["experiment_name"] = f"best_candidate_com_fundo_seed{seed}"
    job["seed"] = seed
    return job


def _render(jobs: list[dict[str, Any]], source: Path) -> str:
    lines = [
        "# Expansao estatistica do Artigo 1 RGB: 4 seeds existentes + 6 novas = 10.",
        "#",
        f"# Fonte das receitas: {source.as_posix()}",
        "# Novas seeds: 7, 123, 2024, 31337, 777, 555.",
        "# Ordem: tres arquiteturas lideres primeiro; EfficientNetB0/B2 depois.",
        "#",
        "# Execucao completa:",
        "#   uv run python -m backend.train_pipeline --config backend/training_jobs_rgb_seed_expansion_10.toml",
        "# Somente lideres:",
        "#   uv run python -m backend.train_pipeline --config backend/training_jobs_rgb_seed_expansion_10.toml --tag leaders-first",
        "",
    ]
    preferred_order = [
        "id", "enabled", "tags", "notes", "stat_group", "stat_repeat", "stat_seed",
        "stat_total_repeats", "source_job_id", "dataset_name", "data_path", "model_name",
        "experiment_name",
    ]
    for job in jobs:
        lines.append("[[jobs]]")
        emitted: set[str] = set()
        for key in preferred_order:
            if key in job:
                lines.append(f"{key} = {_toml_value(job[key])}")
                emitted.add(key)
        for key, value in job.items():
            if key not in emitted:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()
    source_jobs = _source_jobs(args.source)
    jobs = [
        _expanded_job(source_jobs[model], model, seed, repeat)
        for model in MODEL_ORDER
        for repeat, seed in enumerate(NEW_SEEDS, start=len(EXISTING_SEEDS) + 1)
    ]
    args.output.write_text(_render(jobs, args.source), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
