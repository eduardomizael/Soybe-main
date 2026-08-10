"""Gera manifestos e jobs das novas baterias RGB 8, 9 e 11."""
from __future__ import annotations

import tomllib
import csv
import random
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "novas_baterias_stages"
MANIFEST = "docs/bateria_final_eduardo/splits_agrupados_10sementes.csv"
PARTITIONS = (42, 1337, 2026, 9001, 7)
TRAIN_SEEDS = (101, 202, 303, 404)
CANONICAL_SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)


def load_jobs(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return tomllib.load(f)["jobs"]


def quote(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(quote(v) for v in value) + "]"
    return str(value)


def write_jobs(path: Path, jobs: list[dict], title: str) -> None:
    lines = [f"# {title}", ""]
    for job in jobs:
        lines += ["[[jobs]]"]
        lines += [f"{key} = {quote(value)}" for key, value in job.items()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_block8_manifests() -> dict[str, Path]:
    source = ROOT / "docs/bateria_final_eduardo/splits_agrupados_10sementes.csv"
    out = ROOT / "manifestos_bloco8"
    out.mkdir(exist_ok=True)
    inventory: dict[str, list[Path]] = defaultdict(list)
    for class_dir in (ROOT / "data/com_fundo").iterdir():
        if class_dir.is_dir():
            for image in class_dir.glob("grao_DSC_*_id*.png"):
                stem = image.name.split("_id", 1)[0].replace("grao_", "")
                inventory[stem].append(image.resolve())
    for images in inventory.values():
        images.sort(key=lambda p: int(p.stem.rsplit("_id", 1)[1]))

    assignments: dict[int, dict[str, str]] = defaultdict(dict)
    with source.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            assignments[int(row["seed"])][row["dsc"]] = row["subset"]

    result: dict[str, Path] = {}
    for seed in CANONICAL_SEEDS:
        rng = random.Random(seed)
        train, val, test, pool = [], [], [], []
        for dsc, subset in assignments[seed].items():
            grains = inventory.get(dsc, [])
            if subset == "train":
                train.extend(grains)
            elif subset == "val":
                val.extend(grains)
            elif subset == "test":
                order = list(grains)
                rng.shuffle(order)
                middle = len(order) // 2
                test.extend(order[:middle])
                pool.extend(order[middle:])
        by_class = defaultdict(list)
        for image in train:
            by_class[image.parent.name].append(image)
        pool_by_class = Counter(image.parent.name for image in pool)
        clean_train = list(train)
        leaked_train = []
        accepted_pool = []
        for class_name, images in by_class.items():
            k = min(pool_by_class[class_name], max(0, len(images) - 1))
            rng.shuffle(images)
            leaked_train.extend(images[k:])
            class_pool = [image for image in pool if image.parent.name == class_name]
            rng.shuffle(class_pool)
            accepted_pool.extend(class_pool[:k])
        leaked_train.extend(accepted_pool)

        for condition, train_images in (("limpa", clean_train), ("vazada", leaked_train)):
            path = out / f"bloco8_seed{seed}_{condition}.csv"
            temp_path = path.with_suffix(".csv.tmp")
            with temp_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["caminho_arquivo", "classe", "subset"])
                for subset, images in (("train", train_images), ("val", val), ("test", test)):
                    for image in images:
                        writer.writerow([str(image), image.parent.name, subset])
            temp_path.replace(path)
            result[f"{seed}_{condition}"] = path
    return result


def main() -> None:
    source = load_jobs(ROOT / "backend/training_jobs_estatistic_analisis.toml")
    source += load_jobs(ROOT / "backend/training_jobs_bateria_final_eduardo.toml")
    by_model_seed = {
        (j["model_name"], int(j["seed"])): j
        for j in source
        if j.get("dataset_name") == "com_fundo"
    }
    OUT.mkdir(parents=True, exist_ok=True)

    manifests = generate_block8_manifests()
    block8: list[dict] = []
    for model in ("ResNet50", "MobileNetV3"):
        for seed in CANONICAL_SEEDS:
            base = deepcopy(by_model_seed[(model, seed)])
            for condition in ("limpa", "vazada"):
                job = deepcopy(base)
                job.update(
                    id=f"vazamento_teste_fixo_{model.lower()}_seed{seed}_{condition}",
                    experiment_name="vazamento_teste_fixo",
                    seed=seed,
                    split_strategy="predefined",
                    split_manifest=str(manifests[f"{seed}_{condition}"]),
                    dataset_name="com_fundo",
                    data_path="data/com_fundo",
                    notes=f"Bloco 8: condição {condition}; teste fixo; semente {seed}; modelo {model}.",
                    tags=["artigo1", "bloco8", condition],
                )
                block8.append(job)

    block9: list[dict] = []
    for model in ("ResNet50", "MobileNetV3"):
        for partition in PARTITIONS:
            base = deepcopy(by_model_seed[(model, partition)])
            for train_seed in TRAIN_SEEDS:
                job = deepcopy(base)
                job.update(
                    id=f"particao_x_semente_{model.lower()}_part{partition}_seed{train_seed}",
                    experiment_name="particao_x_semente",
                    seed=train_seed,
                    partition_seed=partition,
                    split_strategy="grouped",
                    split_manifest=MANIFEST,
                    dataset_name="com_fundo",
                    data_path="data/com_fundo",
                    notes=(
                        f"Bloco 9: partição fixa {partition}; semente de treino "
                        f"{train_seed}; modelo {model}."
                    ),
                    tags=["artigo1", "bloco9", f"particao-{partition}", f"treino-{train_seed}"],
                )
                block9.append(job)

    block11: list[dict] = []
    for model in ("EfficientNetB2", "EfficientNetB3"):
        for seed in CANONICAL_SEEDS:
            base = deepcopy(by_model_seed[(model, seed)])
            base.update(
                id=f"resolucao_224_{model.lower()}_seed{seed}",
                experiment_name="resolucao_224",
                seed=seed,
                input_size=224,
                split_strategy="grouped",
                split_manifest=MANIFEST,
                dataset_name="com_fundo",
                data_path="data/com_fundo",
                notes=f"Bloco 11: {model} em 224 pixels; semente {seed}.",
                tags=["artigo1", "bloco11", "resolucao-224"],
            )
            block11.append(base)

    write_jobs(OUT / "bloco8.toml", block8, "Bloco 8 — vazamento com teste fixo")
    write_jobs(OUT / "bloco9.toml", block9, "Bloco 9 — partição x semente")
    write_jobs(OUT / "bloco11.toml", block11, "Bloco 11 — resolução 224")
    (OUT / "README.txt").write_text(
        f"Jobs gerados: bloco8={len(block8)}, bloco9={len(block9)}, bloco11={len(block11)}.\n",
        encoding="utf-8",
    )
    print(f"Gerados: bloco8={len(block8)}, bloco9={len(block9)}, bloco11={len(block11)}")


if __name__ == "__main__":
    main()
