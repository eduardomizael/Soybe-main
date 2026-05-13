# Backend

API FastAPI responsavel por inferencia, treinamento, historico de modelos e comunicacao em tempo real com o frontend.

## Arquivos Principais

- `main.py`: inicializa o app, configura CORS e registra rotas.
- `schemas.py`: contratos Pydantic usados pela API.
- `routes/inference_routes.py`: carregamento de pesos, transformacoes e inferencia por arquitetura.
- `routes/training_routes.py`: endpoints REST e WebSocket do treinamento.
- `services/inference_service.py`: entrada de negocio para inferencia.
- `services/training_service.py`: motor de treinamento, avaliacao e persistencia de historico.
- `train_pipeline.py`: execucao de treinamentos em lote via CLI.

## Executar API

Na raiz do projeto:

```bash
.\.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

Em Linux/macOS:

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

## Endpoints

### Geral

- `GET /home`: health check.
- `POST /inferencia`: classifica uma ou mais imagens.

`POST /inferencia` recebe `multipart/form-data`:

- `model_name`: arquitetura, por exemplo `EfficientNetB3`;
- `weight_filename`: opcional, nome do `.pth` dentro de `models/`;
- `files`: lista de imagens.

### Treinamento

Prefixo: `/training`

- `GET /models`: lista arquiteturas disponiveis.
- `GET /datasets`: lista datasets detectados em `data/`.
- `GET /status`: retorna estado do treinamento atual.
- `GET /history`: retorna historico salvo em `models/training_history.json`.
- `GET /model_versions/{model_name}`: lista pesos `.pth` existentes para uma arquitetura.
- `POST /start`: inicia treinamento.
- `POST /pause`: pausa treinamento.
- `POST /resume`: retoma treinamento.
- `POST /stop_early`: finaliza antecipadamente salvando o melhor checkpoint disponivel.
- `POST /cancel`: cancela treinamento.
- `WS /training/ws`: envia progresso em tempo real para o frontend.

## Modelos

Arquiteturas suportadas:

- `EfficientNetB0`
- `EfficientNetB2`
- `EfficientNetB3`
- `EfficientNetB7`
- `ResNet50`
- `MobileNetV3`

Os pesos sao lidos de `models/` e, para compatibilidade, alguns caminhos legados em `backend/network/models/` tambem sao testados.

## Dataset

O treinamento usa `ImageFolder`. Estrutura esperada:

```text
data/
  dataset_nome/
    classe_a/
      imagem_1.jpg
    classe_b/
      imagem_2.jpg
```

O backend rejeita caminhos de dataset fora de `data/`.

## Recursos Do Treinamento

O servico de treinamento preserva:

- seed fixa;
- pesos por classe;
- `AdamW`;
- scheduler `ReduceLROnPlateau`;
- treino em duas fases com congelamento de backbone;
- gradient accumulation;
- early stopping;
- pausa, retomada, cancelamento e finalizacao antecipada;
- import tardio de metricas pesadas;
- ajuste de `DataLoader` para Windows;
- historico por epoca e metadados de runtime.

## Pipeline CLI

Para executar treinamentos em sequencia:

```bash
python -m backend.train_pipeline
```

Edite a lista `PIPELINE` em `backend/train_pipeline.py` antes de rodar.

Artefatos gerados em `models/`:

- `.pth` do modelo;
- relatorio `.txt` por treinamento;
- `*_error.txt` em falhas;
- `pipeline_summary_<timestamp>.txt`.

## Validacao

```bash
python -m compileall backend
```
