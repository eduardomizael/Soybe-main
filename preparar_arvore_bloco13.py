"""Bloco 13 - passo 1: cria data/com_fundo_filtrado (48.039 instancias) a partir
de data/com_fundo, excluindo os arquivos de DSC_0320-0323, e VALIDA as contagens
contra a Tabela 2 do artigo. Aborta com erro se qualquer numero divergir.

Rodar da raiz do Soybe-main:  uv run python preparar_arvore_bloco13.py
"""
import re
import shutil
import sys
from pathlib import Path

SRC = Path("data/com_fundo")
DST = Path("data/com_fundo_filtrado")
AMB = re.compile(r"DSC_032[0-3]_")

ESPERADO = {
    "RGB_JPG_Normais": 19769, "RGB_JPG_IMPUREZAS": 14348, "RGB_JPG_QUEBRADOS": 6584,
    "RGB_JPG_CHOCHOS": 4955, "RGB_JPG_ESVERDEADO": 1192, "RGB_JPG_MAMONAS": 553,
    "RGB_JPG_PURPURAS": 398, "RGB_JPG_INSETOS": 240,
}  # total 48.039

def main() -> None:
    if not SRC.is_dir():
        sys.exit(f"ERRO: {SRC} nao encontrado. Rode da raiz do Soybe-main.")
    if DST.exists():
        print(f"{DST} ja existe - validando apenas (apague a pasta para recriar).")
    else:
        print(f"Copiando {SRC} -> {DST} (sem DSC_0320-0323)...")
        for cls_dir in sorted(SRC.iterdir()):
            if not cls_dir.is_dir():
                continue
            out = DST / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.iterdir():
                if f.is_file() and not AMB.search(f.name):
                    shutil.copy2(f, out / f.name)
            print(f"  {cls_dir.name} ok")

    print("\nValidacao contra a Tabela 2 do artigo:")
    erros, total = 0, 0
    for cls, esperado in ESPERADO.items():
        n = sum(1 for f in (DST / cls).iterdir() if f.is_file()) if (DST / cls).is_dir() else 0
        total += n
        ok = n == esperado
        print(f"  {cls:22s} {n:6d}  (esperado {esperado:6d})  {'OK' if ok else 'ERRO'}")
        erros += 0 if ok else 1
    print(f"  TOTAL{'':19s}{total:6d}  (esperado  48039)  {'OK' if total == 48039 else 'ERRO'}")
    extras = [f.name for c in DST.iterdir() if c.is_dir() for f in c.iterdir() if AMB.search(f.name)]
    if extras:
        print(f"  ERRO: {len(extras)} arquivos DSC_0320-0323 dentro da arvore filtrada!")
        erros += 1
    if erros or total != 48039:
        sys.exit("\nNAO PROSSEGUIR - contagens divergem. Mandar esta saida para o Leonardo.")
    print("\nArvore validada. Pode rodar o pre-voo (bloco13_prevoo.toml).")

if __name__ == "__main__":
    main()
