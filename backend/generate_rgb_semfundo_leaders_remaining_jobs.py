from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend" / "training_jobs_rgb_sem_fundo_seed_expansion_10.toml"
OUTPUT = ROOT / "backend" / "training_jobs_rgb_semfundo_leaders_remaining_10.toml"
REMAINING = {
    "ResNet50": {123, 2024, 31337, 777, 555},
    "MobileNetV3": {7, 123, 2024, 31337, 777, 555},
}


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return str(value)


def main() -> int:
    with SOURCE.open("rb") as file:
        source_jobs = tomllib.load(file)["jobs"]
    jobs: list[dict[str, Any]] = []
    for job in source_jobs:
        model = job.get("model_name")
        seed = job.get("seed")
        if model not in REMAINING or seed not in REMAINING[model]:
            continue
        item = deepcopy(job)
        item["tags"] = list(dict.fromkeys([*item.get("tags", []), "semfundo-leaders-10"]))
        item["notes"] = (
            "Completa sem_fundo em 10 seeds para comparação com_fundo por replicação pareada; "
            "mantém integralmente a receita best_candidate."
        )
        jobs.append(item)
    if len(jobs) != 11:
        raise ValueError(f"Esperados 11 jobs; encontrados {len(jobs)}.")

    lines = [
        "# sem_fundo: somente ResNet50 e MobileNetV3, 11 jobs restantes.",
        "# Reutiliza quatro seeds históricas de cada arquitetura e ResNet50/seed7 já concluída.",
        "# Não retomar a fila anterior de 30 jobs para esta etapa.",
        "",
    ]
    for job in jobs:
        lines.append("[[jobs]]")
        for key, value in job.items():
            lines.append(f"{key} = {_value(value)}")
        lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Gerados {len(jobs)} jobs em {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
