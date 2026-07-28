"""Gera as duas filas do protocolo uniforme do Artigo 1 RGB.

Não altera receitas históricas: os overrides de pesos e resolução ficam no TOML
de cada novo experimento, com IDs e experiment_name próprios.
"""

from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_estatistic_analisis.toml"
EXPANSION_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_rgb_seed_expansion_10.toml"
CONVNEXT_PATH = WORKSPACE_ROOT / "backend" / "training_jobs_convnext_tiny_com_fundo_10_seeds.toml"
PRETRAIN_OUTPUT = WORKSPACE_ROOT / "backend" / "training_jobs_rgb_pretrain_v1_10_seeds.toml"
UNIFORM_OUTPUT = WORKSPACE_ROOT / "backend" / "training_jobs_rgb_uniform_protocol_10_seeds.toml"
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)
MODELS = ("ResNet50", "MobileNetV3", "EfficientNetB0", "EfficientNetB2", "EfficientNetB3", "ConvNeXtTiny")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as file:
        return tomllib.load(file)["jobs"]


def _canonical_recipes() -> dict[str, dict[int, dict[str, Any]]]:
    recipes: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in MODELS}
    for path in (HISTORICAL_PATH, EXPANSION_PATH, CONVNEXT_PATH):
        for job in _load_jobs(path):
            model = job.get("model_name")
            seed = job.get("seed")
            if model in recipes and job.get("dataset_name") == "com_fundo" and seed in SEEDS:
                recipes[model][int(seed)] = deepcopy(job)
    missing = {
        model: [seed for seed in SEEDS if seed not in by_seed]
        for model, by_seed in recipes.items()
        if len(by_seed) != len(SEEDS)
    }
    if missing:
        raise RuntimeError(f"Receitas canônicas incompletas: {missing}")
    return recipes


def _write(path: Path, title: str, jobs: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", "# Gerado por backend/generate_rgb_uniform_protocol_jobs.py.", ""]
    for job in jobs:
        lines.append("[[jobs]]")
        for key, value in job.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _pretrain_v1_job(source: dict[str, Any], repeat: int) -> dict[str, Any]:
    job = deepcopy(source)
    model_slug = str(job["model_name"]).lower()
    seed = int(job["seed"])
    job.update(
        {
            "id": f"{model_slug}_com_fundo_pretrain_v1_seed{seed}",
            "tags": ["rgb", "com_fundo", "pretrain-v1", "statistical-repeat", f"seed-{seed}", f"repeat-{repeat:02d}"],
            "notes": "Controle de pré-treino V1: mantém a receita histórica da arquitetura e força IMAGENET1K_V1.",
            "stat_group": f"{model_slug}_com_fundo_pretrain_v1",
            "stat_repeat": repeat,
            "stat_seed": seed,
            "stat_total_repeats": len(SEEDS),
            "source_job_id": str(source["id"]),
            "experiment_name": "pretrain_v1",
            "pretrained_weights": "IMAGENET1K_V1",
        }
    )
    return job


def _uniform_job(source: dict[str, Any], repeat: int) -> dict[str, Any]:
    job = deepcopy(source)
    model_slug = str(job["model_name"]).lower()
    seed = int(job["seed"])
    job.update(
        {
            "id": f"{model_slug}_com_fundo_uniform_protocol_seed{seed}",
            "tags": ["rgb", "com_fundo", "uniform-protocol", "statistical-repeat", f"seed-{seed}", f"repeat-{repeat:02d}"],
            "notes": "Protocolo uniforme RGB: somente a arquitetura varia; todos os demais parâmetros são idênticos.",
            "stat_group": f"{model_slug}_com_fundo_uniform_protocol",
            "stat_repeat": repeat,
            "stat_seed": seed,
            "stat_total_repeats": len(SEEDS),
            "source_job_id": str(source["id"]),
            "experiment_name": "uniform_protocol",
            "pretrained_weights": "IMAGENET1K_V1",
            "input_size": 224,
            "batch_size": 16,
            "num_epochs": 30,
            "learning_rate": 0.0001,
            "fine_tune_learning_rate": 0.00006,
            "early_stopping": True,
            "patience": 7,
            "split_strategy": "stratified",
            "checkpoint_metric": "val_macro_f1",
            "sampler_strategy": "weighted",
            "loss_name": "cross_entropy",
            "class_weight_strategy": "sqrt_inverse",
            "optimizer_name": "AdamW",
            "weight_decay": 0.0001,
            "scheduler_name": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
            "freeze_backbone_epochs": 2,
        }
    )
    return job


def main() -> int:
    recipes = _canonical_recipes()
    pretrain_jobs = [
        _pretrain_v1_job(recipes[model][seed], repeat)
        for model in ("ResNet50", "MobileNetV3")
        for repeat, seed in enumerate(SEEDS, start=1)
    ]
    uniform_jobs = [
        _uniform_job(recipes[model][seed], repeat)
        for model in MODELS
        for repeat, seed in enumerate(SEEDS, start=1)
    ]
    _write(PRETRAIN_OUTPUT, "Experimento A: pesos IMAGENET1K_V1 em ResNet50 e MobileNetV3 (20 jobs).", pretrain_jobs)
    _write(UNIFORM_OUTPUT, "Experimento B: protocolo uniforme RGB (6 arquiteturas x 10 seeds = 60 jobs).", uniform_jobs)
    print(f"Gerados {len(pretrain_jobs)} jobs em {PRETRAIN_OUTPUT}")
    print(f"Gerados {len(uniform_jobs)} jobs em {UNIFORM_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
