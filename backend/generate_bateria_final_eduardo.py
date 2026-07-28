"""Gera a bateria final RGB agrupada (180 execuções) sem iniciar treinamento."""
from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "training_jobs_bateria_final_eduardo.toml"
STAGE_DIR = ROOT / "backend" / "bateria_final_stages"
MANIFEST = "docs/bateria_final_eduardo/splits_agrupados_10sementes.csv"
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)
MODELS = ("EfficientNetB0", "EfficientNetB2", "EfficientNetB3", "MobileNetV3", "ResNet50", "ConvNeXtTiny")


def load_jobs(path: Path):
    with path.open("rb") as f:
        return tomllib.load(f)["jobs"]


def toml(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list): return "[" + ", ".join(toml(x) for x in v) + "]"
    return str(v)


def main():
    sources = load_jobs(ROOT / "backend/training_jobs_estatistic_analisis.toml")
    sources += load_jobs(ROOT / "backend/training_jobs_rgb_seed_expansion_10.toml")
    sources += load_jobs(ROOT / "backend/training_jobs_convnext_tiny_com_fundo_10_seeds.toml")
    by_key = {(j["model_name"], int(j["seed"],)): j for j in sources if j.get("dataset_name") == "com_fundo"}
    sem_sources = load_jobs(ROOT / "backend/training_jobs_rgb_sem_fundo_seed_expansion_10.toml")
    sem_by_key = {(j["model_name"], int(j["seed"])): j for j in sem_sources}
    jobs = []

    def base(model, seed, dataset="com_fundo"):
        source = sem_by_key.get((model, seed), by_key[(model, seed)]) if dataset == "sem_fundo" else by_key[(model, seed)]
        j = deepcopy(source)
        j.update({"split_strategy": "grouped", "split_manifest": MANIFEST, "num_epochs": 30,
                  "checkpoint_metric": "val_macro_f1", "sampler_strategy": "weighted",
                  "class_weight_strategy": "sqrt_inverse", "optimizer_name": "AdamW",
                  "scheduler_name": "ReduceLROnPlateau", "scheduler_factor": 0.5,
                  "scheduler_patience": 2, "scheduler_min_lr": 1e-6,
                  "freeze_backbone_epochs": 2, "early_stopping": True, "patience": 7,
                  "dataset_name": dataset, "data_path": f"data/{dataset}", "seed": seed})
        return j

    def add(j, exp, recipe, index, **overrides):
        j.update(overrides)
        j["id"] = f"{exp}_{j['model_name'].lower()}_seed{j['seed']}" + (f"_{index}" if index else "")
        j["experiment_name"], j["split_protocol"], j["recipe"] = exp, "grouped", recipe
        j["tags"] = ["bateria-final-eduardo", exp, f"seed-{j['seed']}"]
        jobs.append(j)

    for model in MODELS:
        for seed in SEEDS: add(base(model, seed), "benchmark_agrupado", "per_arch", 0)
    for model in ("ResNet50", "MobileNetV3"):
        for seed in SEEDS: add(base(model, seed, "sem_fundo"), "semfundo_agrupado", "per_arch", 0)
    for seed in SEEDS:
        add(base("ResNet50", seed), "swin_t", "per_arch", 0, model_name="SwinT", pretrained_weights="IMAGENET1K_V1", input_size=224, batch_size=16, learning_rate=1e-4, fine_tune_learning_rate=6e-5)
    for model in ("MobileNetV3", "EfficientNetB0"):
        for seed in SEEDS: add(base(model, seed), "uniforme", "uniform_224", 0, input_size=224, batch_size=16, learning_rate=1e-4, fine_tune_learning_rate=6e-5, weight_decay=1.5e-4)
    factors = ("split_strategy", "sampler_strategy", "optimizer_name", "scheduler_name", "freeze_backbone_epochs", "checkpoint_metric", "num_epochs")
    for factor in factors:
        for seed in SEEDS:
            add(base("MobileNetV3", seed), f"ablacao_{factor}", "per_arch", 0, ablation_factor=factor, ablation_role="factor")
    lines = ["# Bateria final Eduardo — 180 execuções; gerado sem executar treinos.", ""]
    for job in jobs:
        lines.append("[[jobs]]")
        for key, value in job.items(): lines.append(f"{key} = {toml(value)}")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    STAGE_DIR.mkdir(exist_ok=True)
    stage_jobs = {
        "piloto": [j for j in jobs if j["experiment_name"] == "benchmark_agrupado" and j["model_name"] == "ResNet50" and j["seed"] == 42],
        "bloco1": [j for j in jobs if j["experiment_name"] == "benchmark_agrupado"],
        "bloco3": [j for j in jobs if j["experiment_name"] == "swin_t"],
        "bloco4": [j for j in jobs if j["experiment_name"] == "uniforme"],
        "bloco2": [j for j in jobs if j["experiment_name"] == "semfundo_agrupado"],
        "bloco5": [j for j in jobs if j["experiment_name"].startswith("ablacao_")],
    }
    for stage, selected in stage_jobs.items():
        stage_lines = [f"# {stage}: gerado pelo gerador da bateria final.", ""]
        for job in selected:
            stage_lines.append("[[jobs]]")
            for key, value in job.items(): stage_lines.append(f"{key} = {toml(value)}")
            stage_lines.append("")
        (STAGE_DIR / f"{stage}.toml").write_text("\n".join(stage_lines), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {OUT}")
    print("Estágios: " + ", ".join(f"{k}={len(v)}" for k, v in stage_jobs.items()))


if __name__ == "__main__": main()
