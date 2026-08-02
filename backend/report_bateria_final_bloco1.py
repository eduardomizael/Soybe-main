"""Gera o relatório textual consolidado do Bloco 1 a partir da análise CSV."""
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "models/bateria_final_bloco1_analysis_20260801"
OUT = ANALYSIS / "relatorio_consolidado_bloco1.md"

def read(name):
    with (ANALYSIS / name).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def main():
    meta = json.loads((ANALYSIS / "analysis_metadata.json").read_text(encoding="utf-8"))
    summary, tests = read("rgb_seed_summary.csv"), read("rgb_seed_statistical_tests.csv")
    lines = ["# Relatório consolidado — Bloco 1", "", f"Execuções válidas: {meta['found_runs']}/{meta['expected_runs']}", "Split: `grouped` · conjunto: `com_fundo` · sementes: 10", "", "## Resumo por arquitetura", "", "| Arquitetura | n | Accuracy média | Accuracy DP | Macro-F1 média | Macro-F1 DP | IC95% Macro-F1 |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['arquitetura']} | {row['n']} | {float(row['accuracy_media']):.2f}% | {float(row['accuracy_desvio_padrao']):.2f} | {float(row['macro_f1_media']):.2f}% | {float(row['macro_f1_desvio_padrao']):.2f} | {float(row['macro_f1_ic95_inferior']):.2f}%–{float(row['macro_f1_ic95_superior']):.2f}% |")
    lines += ["", "## Testes estatísticos", "", "Os testes e p-valores completos estão em `rgb_seed_statistical_tests.csv`; a interpretação deve considerar a correção de Holm e o fato de serem 10 sementes.", ""]
    for row in tests: lines.append(f"- {row.get('teste','')} / {row.get('metrica','')} — {row.get('arquitetura_a','')} vs {row.get('arquitetura_b','')}: p={row.get('p_valor','')}, p-Holm={row.get('p_holm','')}")
    lines += ["", "## Artefatos", "", "Cada execução possui checkpoint, relatório `.txt` e CSV de predições com `caminho_arquivo, classe_verdadeira, classe_predita`. O manifesto usado está incluído no pacote.", "", "## Próximo passo", "", "Após revisar este relatório e aceitar o Bloco 1, executar o Bloco 3 (Swin-T), conforme a ordem do `00_LEIA_PRIMEIRO.md`.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)

if __name__ == "__main__": main()
