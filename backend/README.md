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
- split configuravel entre `random` e `stratified`;
- sampler configuravel entre `shuffle` e `weighted`;
- criterio de checkpoint configuravel entre `val_loss`, `val_accuracy` e `val_macro_f1`;
- treino em duas fases com congelamento de backbone;
- gradient accumulation;
- early stopping configuravel por job;
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

Para forcar um job a rodar ate `num_epochs`, use:

```python
"early_stopping": False
```

Mesmo com early stopping desativado, o servico continua salvando o melhor checkpoint pelo `checkpoint_metric` configurado.

Para comparar modelos com uma configuracao mais adequada a datasets desbalanceados, use:

```python
"split_strategy": "stratified",
"checkpoint_metric": "val_macro_f1",
"sampler_strategy": "weighted"
```

Com `checkpoint_metric: "val_macro_f1"`, o melhor checkpoint passa a ser escolhido por macro F1 de validacao. O `val_loss` continua sendo registrado e usado pelo scheduler.

Os resultados finais incluem o bloco `efficiency` com throughput de treino, throughput de teste, contagem de parametros e tamanho do checkpoint.

A pipeline CLI gera automaticamente dois experimentos por modelo candidato:

- `baseline`: `random` + `shuffle` + checkpoint por `val_loss` + early stopping.
- `experimental`: `stratified` + `weighted` + checkpoint por `val_macro_f1` + treino ate o fim.

Modelos candidatos padrao:

- `MobileNetV3`
- `EfficientNetB0`
- `EfficientNetB2`
- `EfficientNetB3`

`ResNet50` e `EfficientNetB7` permanecem parametrizados no arquivo, mas ficam fora da comparacao padrao por custo/beneficio inferior nos relatorios atuais.

Artefatos gerados em `models/`:

- `.pth` do modelo;
- relatorio `.txt` por treinamento;
- `*_error.txt` em falhas;
- `pipeline_summary_<timestamp>.txt`.

## Validacao

```bash
python -m compileall backend
```
