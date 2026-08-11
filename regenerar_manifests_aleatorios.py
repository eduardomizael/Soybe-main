"""Regenerate the random-protocol (per-grain, class-stratified) partition
manifests deterministically, for the ten canonical seeds.

The random-protocol runs used ``split_strategy = "stratified"`` with
train/val = 0.8/0.1 (see configs of the original battery). This script does
NOT reimplement the split: it imports ``_stratified_split`` from the training
engine itself, so the logic is the one that produced the reported runs.

Exact reproduction additionally requires:
  * the same dataset tree the runs saw (the curated training set — 48,039
    instances after the 934 exclusions — NOT the full released 48,973);
  * the pinned environment (`uv sync --frozen`): numpy 2.2.6 fixes the RNG
    stream, torchvision 0.23.0 fixes the ImageFolder file ordering (sorted
    class dirs, sorted filenames).

Usage:
    uv run scripts/regenerar_manifests_aleatorios.py --data-dir <com_fundo_tree> \
        [--expect-total 48039] [--out-dir manifests/random]

Output: one CSV per seed (random_seed_<S>.csv) with columns
    semente, subset, classe, arquivo
sorted by (subset, classe, arquivo). Membership — not row order — defines the
partition; the engine shuffles feeding order at train time independently.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from torchvision.datasets import ImageFolder

try:  # public repository layout
    from soybean_bench.training.training_service import _stratified_split
except ImportError:  # Eduardo's development repository (Soybe-main, branch bateria-final)
    from backend.services.training_service import _stratified_split

SEEDS = [42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555]
TRAIN_SPLIT, VAL_SPLIT = 0.8, 0.1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="root of the curated com_fundo tree (one subdir per class)")
    ap.add_argument("--out-dir", default=Path("manifests/random"), type=Path)
    ap.add_argument("--expect-total", type=int, default=None,
                    help="abort if the dataset instance count differs (e.g. 48039)")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()

    dataset = ImageFolder(str(args.data_dir))
    total = len(dataset.samples)
    per_class = {cls: sum(1 for _, t in dataset.samples if t == i)
                 for i, cls in enumerate(dataset.classes)}
    print(f"dataset: {total} instances, {len(dataset.classes)} classes")
    for cls, n in per_class.items():
        print(f"  {cls}: {n}")
    if args.expect_total is not None and total != args.expect_total:
        raise SystemExit(
            f"ABORT: {total} instances found, expected {args.expect_total}. "
            "This is not the tree the runs saw; the regenerated splits would "
            "not match the reported ones.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(dataset.root)
    for seed in args.seeds:
        train_ds, val_ds, test_ds = _stratified_split(
            dataset, len(dataset.classes), TRAIN_SPLIT, VAL_SPLIT, seed)
        rows = []
        for subset_name, subset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            for idx in subset.indices:
                path, target = dataset.samples[idx]
                rows.append((seed, subset_name, dataset.classes[target],
                             Path(path).relative_to(root).as_posix()))
        rows.sort(key=lambda r: (r[1], r[2], r[3]))
        out = args.out_dir / f"random_seed_{seed}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["semente", "subset", "classe", "arquivo"])
            w.writerows(rows)
        n_tr = sum(1 for r in rows if r[1] == "train")
        n_va = sum(1 for r in rows if r[1] == "val")
        n_te = sum(1 for r in rows if r[1] == "test")
        print(f"seed {seed}: train {n_tr} | val {n_va} | test {n_te} -> {out}")

    print("\nDone. If Eduardo's original in-run splits surface later, diff the "
          "memberships; any divergence means dataset tree or environment differs.")


if __name__ == "__main__":
    main()
