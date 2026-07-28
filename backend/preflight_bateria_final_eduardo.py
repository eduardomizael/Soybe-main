"""Pré-voo obrigatório da bateria final; não inicia treinamento."""
from __future__ import annotations
import csv, hashlib, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/bateria_final_eduardo/splits_agrupados_10sementes.csv"
SEEDS = (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def key(path: Path):
    m = re.search(r"(DSC_\d+)", path.stem)
    return (path.parent.name.replace("_SF", ""), m.group(1)) if m else None

def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def main():
    assignments = defaultdict(dict)
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f): assignments[int(row["seed"])][(row["classe"], row["dsc"])] = row["subset"]
    if set(assignments) != set(SEEDS): raise SystemExit("Manifesto não contém exatamente as 10 sementes exigidas.")
    for dataset_name in ("com_fundo", "sem_fundo"):
        root = ROOT / "data" / dataset_name
        samples = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
        classes = sorted({p.parent.name.replace("_SF", "") for p in samples})
        if len(classes) != 8: raise SystemExit(f"{dataset_name}: esperadas 8 classes, encontradas {len(classes)}")
        for seed in SEEDS:
            parts = defaultdict(set); missing = []
            for p in samples:
                k = key(p); subset = assignments[seed].get(k)
                if subset: parts[subset].add(k)
                elif k and not re.search(r"DSC_032[0-3]", p.stem): missing.append(str(p))
            if missing: raise SystemExit(f"{dataset_name}/seed {seed}: {len(missing)} arquivo(s) fora do manifesto")
            if set(parts) != {"train", "val", "test"}: raise SystemExit(f"{dataset_name}/seed {seed}: subset ausente")
            if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]: raise SystemExit(f"{dataset_name}/seed {seed}: grupos não disjuntos")
            print(f"OK {dataset_name} seed={seed}: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")
    for rel in ("data/com_fundo/RGB_JPG_CHOCHOS/grao_DSC_0320_id1.png", "data/com_fundo/RGB_JPG_Normais/grao_DSC_0320_id1.png"):
        p = ROOT / rel
        print(f"MD5 {rel}: {md5(p) if p.exists() else 'ARQUIVO_AUSENTE'}")
    print("PRÉ-VOO OK. Próxima etapa permitida: piloto ResNet50 seed 42 com_fundo.")

if __name__ == "__main__": main()
