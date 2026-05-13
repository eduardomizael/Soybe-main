# Backend (FastAPI)

API responsavel por inferencia e treinamento.

## Arquivos principais

- main.py: inicializa app, CORS e registra rotas.
- routes/inference_routes.py: carga de modelos e pipeline de classificacao.
- services/inference_service.py: interface de inferencia usada pelo endpoint.
- routes/training_routes.py: REST + WebSocket para treinamento.
- services/training_service.py: motor de treinamento e historico.
- schemas.py: contratos Pydantic.

## Como executar

```bash
source env/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

## Endpoints

### Inference

- POST /inferencia
  - multipart/form-data
  - campos:
    - model_name (obrigatorio)
    - weight_filename (opcional)
    - files (um ou mais arquivos)

- GET /home

### Training

Prefixo /training

- GET /models
- GET /datasets
- GET /status
- POST /start
- POST /pause
- POST /resume
- POST /stop_early
- POST /cancel
- GET /history
- GET /model_versions/{model_name}
- WS /training/ws

## Regras importantes

- Dataset de treino deve estar dentro de data/.
- Pesos e historico sao salvos em models/.
