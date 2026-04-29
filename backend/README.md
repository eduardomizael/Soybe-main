# Backend (FastAPI)

Este diretório contém a API de inferência usada pelo site.

## Estrutura

- `main.py`: inicializa o `FastAPI`, configura CORS e expõe os endpoints (`/inferencia`, `/home`).
- `routes/`: rotas opcionais e testes de inferência (não usado diretamente no fluxo atual).
  - `inference_routes.py`: exemplo de rota e código auxiliar para teste com EfficientNet.
- `schemas/`: modelos de dados da API (Pydantic).
  - `inference_schema.py`: esquemas específicos de inferência.
- `schemas.py`: esquemas principais retornados/recebidos pela API (`InferenceRequest`, `InferenceResponse`).
- `services/`: camada de serviço que contém a lógica de negócio.
  - `inference_service.py`: ponto de entrada para rodar a inferência a partir dos bytes do arquivo.
- `documentation.txt`: notas internas de arquitetura e testes.

## Como executar

1. Instale dependências (usar o ambiente virtual do projeto):

```bash
pip install -r requirements.txt
```

2. Suba o servidor de desenvolvimento:

```bash
uvicorn backend.main:app --reload
```

```
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

3. Endpoints:
- `POST /inferencia`: recebe `model_name` e arquivos de imagem (multipart/form-data) em bytes.
- `GET /home`: verificação de saúde da API

## Pipeline De Treinamento Via Script

Também é possível executar treinamentos em sequência sem subir a API.

1. Edite a lista `PIPELINE` em [backend/train_pipeline.py](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py) com os jobs desejados.
2. Garanta que `data_path` aponte para um diretório dentro de `data/`.
3. Execute:

```bash
python -m backend.train_pipeline
```

O script imprime no terminal:
- status de preparação
- progresso por batch
- métricas por época
- caminho do `.pth` salvo ao final de cada job

O pipeline também suporta:
- `AdamW` com `weight_decay`
- `ReduceLROnPlateau`
- `seed` fixa para reprodutibilidade
- `freeze_backbone_epochs` para treino em duas fases
- `accumulation_steps` para modelos pesados como `EfficientNetB7`

Artefatos gerados em `models/`:
- um `.pth` por treinamento bem-sucedido
- um `.txt` por treinamento com métricas, runtime e histórico por época
- um `*_error.txt` para jobs que falharem
- um `pipeline_summary_<timestamp>.txt` com o resumo consolidado da execução

## Convenções
- Importações internas devem usar o prefixo `backend.` (ex.: `from backend.services.inference_service import run_inference`).
- Evite acessar modelos diretamente nas rotas: sempre passe pela camada `services/`.
- `schemas.py` define contratos claros para entrada/saída.
