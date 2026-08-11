"""Gera os 10 jobs aleatórios do ConvNeXt-Tiny para o Bloco 12."""
from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/training_jobs_convnext_tiny_com_fundo_10_seeds.toml"
OUTPUT = ROOT / "backend/bateria_final_stages/bloco12_convnext_random.toml"
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)


def quote(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(quote(item) for item in value) + "]"
    return str(value)


def main() -> None:
    with SOURCE.open("rb") as file:
        source_jobs = tomllib.load(file)["jobs"]
    by_seed = {int(job["seed"]): job for job in source_jobs}
    jobs = []
    for seed in SEEDS:
        job = deepcopy(by_seed[seed])
        job.update(
            id=f"convnexttiny_com_fundo_random_seed{seed}",
            experiment_name="random_convnext_perclasse",
            split_strategy="stratified",
            split_manifest=None,
            notes=(
                f"Bloco 12: ConvNeXt-Tiny RGB com_fundo, protocolo aleatório, "
                f"seed {seed}; complemento F1 por classe."
            ),
            tags=["artigo1", "bloco12", "convnext", "random", f"seed-{seed}"],
        )
        job.pop("split_manifest", None)
        jobs.append(job)

    lines = ["# Bloco 12 — ConvNeXt-Tiny aleatório — 10 execuções", ""]
    for job in jobs:
        lines.append("[[jobs]]")
        lines.extend(f"{key} = {quote(value)}" for key, value in job.items())
        lines.append("")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {OUTPUT}")


if __name__ == "__main__":
    main()
