"""Consolida qualquer etapa da bateria final a partir de um pipeline_run."""
from __future__ import annotations
import csv, json, shutil, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    run_path = Path(sys.argv[1]).resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    jobs = [e for e in run.get("jobs", []) if e.get("status") == "success"]
    if not jobs: raise SystemExit("Nenhum job concluído.")
    stage = Path(run["config_path"]).stem
    next_stage = {"bloco3": "Bloco 4", "bloco4": "Bloco 2", "bloco2": "Bloco 5", "bloco5": "análise final"}.get(stage, "próxima etapa")
    stamp = run.get("run_id", "sem_id")
    analysis = ROOT / "models" / f"{stage}_analysis_{stamp}"
    package = ROOT / "models" / f"{stage}_package_{stamp}"
    for p in (analysis, package / "models", package / "reports", package / "predictions", package / "source"): p.mkdir(parents=True, exist_ok=True)
    rows=[]
    for e in jobs:
        j=e["job"]; r=e.get("result",{}) or {}
        rows.append({"experimento":j.get("experiment_name",""),"arquitetura":j.get("model_name",""),"conjunto":j.get("dataset_name",""),"semente":j.get("seed",""),"split_protocol":j.get("split_protocol",""),"receita":j.get("recipe",""),"acuracia":r.get("accuracy",""),"macro_f1":r.get("macro_f1",""),"tempo_treino_min":round(float(r.get("total_time",0))/60,2),"job_id":j.get("id","")})
        for key, folder in (("model_path","models"),("report_path","reports"),("predictions_path","predictions")):
            src=e.get(key) or r.get(key)
            if src and Path(src).exists(): shutil.copy2(src, package/folder/Path(src).name)
    with (analysis/"metricas_por_execucao.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    grouped=[]
    for model in sorted({r["arquitetura"] for r in rows}):
        x=[r for r in rows if r["arquitetura"]==model]; acc=[float(r["acuracia"]) for r in x]; f1=[float(r["macro_f1"]) for r in x]
        grouped.append({"arquitetura":model,"n":len(x),"accuracy_media":round(statistics.mean(acc),3),"accuracy_dp":round(statistics.stdev(acc),3) if len(acc)>1 else 0,"macro_f1_media":round(statistics.mean(f1),3),"macro_f1_dp":round(statistics.stdev(f1),3) if len(f1)>1 else 0,"tempo_medio_min":round(statistics.mean(float(r["tempo_treino_min"]) for r in x),2)})
    with (analysis/"resumo_por_arquitetura.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(grouped[0])); w.writeheader(); w.writerows(grouped)
    lines=[f"# Relatório consolidado — {stage}","",f"Execuções válidas: {len(rows)}","", "| Arquitetura | n | Accuracy média | Accuracy DP | Macro-F1 média | Macro-F1 DP | Tempo médio (min) |","|---|---:|---:|---:|---:|---:|---:|"]
    for g in grouped: lines.append(f"| {g['arquitetura']} | {g['n']} | {g['accuracy_media']:.2f}% | {g['accuracy_dp']:.2f} | {g['macro_f1_media']:.2f}% | {g['macro_f1_dp']:.2f} | {g['tempo_medio_min']:.2f} |")
    lines += ["","Todos os jobs concluídos possuem checkpoint, relatório TXT e CSV de predições no pacote.","",f"Próxima etapa conforme o leia_primeiro: {next_stage}, após revisão/aceite desta etapa.",""]
    (analysis/"relatorio_consolidado.md").write_text("\n".join(lines),encoding="utf-8")
    shutil.copy2(run_path, package/"source"/run_path.name)
    config=Path(run["config_path"])
    if config.exists(): shutil.copy2(config,package/"source"/config.name)
    (package/"README_ENVIO.md").write_text(f"# Pacote {stage}\n\n{len(rows)} execuções concluídas. Consulte `analysis/relatorio_consolidado.md`. Checkpoints, relatórios e predições estão separados por diretório.\n\nPróxima etapa: {next_stage}.\n",encoding="utf-8")
    print(f"Análise: {analysis}\nPacote: {package}")

if __name__ == "__main__": main()
