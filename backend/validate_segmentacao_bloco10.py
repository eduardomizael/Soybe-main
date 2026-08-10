"""Prepara e verifica a validação manual da segmentação do Bloco 10.

Uso:
  python backend/validate_segmentacao_bloco10.py prepare
  python backend/validate_segmentacao_bloco10.py check-counts resultados_agrupados/10_segmentacao/contagem_deteccao.csv

O desenho dos 160 polígonos continua deliberadamente manual para evitar viés.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/bateria_final_eduardo/splits_agrupados_10sementes.csv"
DATA = ROOT / "data/com_fundo"
OUT = ROOT / "resultados_agrupados/10_segmentacao"


def prepare() -> None:
    photos: dict[str, set[str]] = defaultdict(set)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            photos[row["classe"]].add(row["dsc"])
    rng = random.Random(20260807)
    selected: list[tuple[str, str]] = []
    for class_name in sorted(photos):
        selected.extend((class_name, dsc) for dsc in rng.sample(sorted(photos[class_name]), min(2, len(photos[class_name]))))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fotos_amostradas.txt").write_text(
        "\n".join(f"{class_name}\t{dsc}" for class_name, dsc in selected) + "\n",
        encoding="utf-8",
    )
    rows = []
    for class_name, dsc in selected:
        for image in sorted((DATA / class_name).glob(f"grao_{dsc}_id*.png")):
            rows.append((str(image.resolve()), class_name, dsc))
    rng.shuffle(rows)
    sampled: list[tuple[str, str, str]] = []
    for class_name in sorted(photos):
        candidates = [row for row in rows if row[1] == class_name]
        sampled.extend(candidates[:20])
    with (OUT / "recortes_amostrados.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["caminho_arquivo", "classe", "dsc"])
        writer.writerows(sampled)
    (OUT / "README_MANUAL.md").write_text(
        "# Bloco 10 — validação manual\n\n"
        "1. Execute a segmentação nas 16 fotografias listadas em fotos_amostradas.txt.\n"
        "2. Preencha contagem_deteccao.csv comparando a imagem original e os recortes.\n"
        "3. Para os 160 recortes de recortes_amostrados.csv, desenhe o polígono no LabelMe "
        "antes de abrir a máscara automática.\n"
        "4. Salve os JSON em delineamento_manual/ e as máscaras binárias em mascaras_automaticas/.\n"
        "5. Rode check-counts para verificar o fechamento das contagens.\n",
        encoding="utf-8",
    )
    print(f"Fotos: {len(selected)}; recortes amostrados: {len(sampled)}; saída: {OUT}")


def check_counts(path: Path) -> None:
    required = {"graos_visiveis", "deteccoes", "corretas", "espurias", "truncadas", "fundidas", "perdidas"}
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise SystemExit(f"Colunas ausentes: {', '.join(sorted(missing))}")
    errors = []
    for row in rows:
        n = {key: int(row[key]) for key in required}
        if n["corretas"] + n["espurias"] + n["truncadas"] + n["fundidas"] != n["deteccoes"]:
            errors.append(f"{row.get('dsc')}: deteccoes não fecha")
        if "graos_em_fundidas" in row and row["graos_em_fundidas"]:
            fused_grains = int(row["graos_em_fundidas"])
            if n["corretas"] + n["truncadas"] + n["perdidas"] + fused_grains != n["graos_visiveis"]:
                errors.append(f"{row.get('dsc')}: graos_visiveis não fecha")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(rows)} fotografias com contagens fechadas.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check-counts"))
    parser.add_argument("csv_path", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if not args.csv_path:
            raise SystemExit("Informe o caminho de contagem_deteccao.csv.")
        check_counts(Path(args.csv_path))


if __name__ == "__main__":
    main()
