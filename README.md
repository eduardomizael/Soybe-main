# Soybe

Sistema web para classificacao e treinamento de modelos de visao computacional aplicados a graos de soja. O projeto combina uma API em FastAPI com uma interface React/Vite para inferencia, dashboards de resultados e operacao de treinamento.

## Visao Geral

O Soybe permite:

- enviar uma imagem ou um lote de imagens para classificacao;
- selecionar arquitetura e versao de pesos treinados;
- visualizar resultados individuais de inferencia;
- acompanhar um dashboard agregado do lote processado;
- iniciar, pausar, retomar, cancelar ou finalizar antecipadamente treinamentos;
- consultar historico de treinamentos, metricas por classe, matriz de confusao e curvas ROC quando disponiveis;
- executar pipelines de treinamento por script, sem depender da interface web.

## Estrutura

```text
Soybe-main/
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── routes/
│   │   ├── inference_routes.py
│   │   └── training_routes.py
│   ├── services/
│   │   ├── inference_service.py
│   │   └── training_service.py
│   └── train_pipeline.py
├── frontend/
│   ├── src/app/App.tsx
│   ├── src/app/components/
│   ├── package.json
│   └── vite.config.ts
├── data/
├── models/
├── src/models/
├── requirements.txt
└── README.md
```

`data/` e `models/` sao usados para dados locais e pesos gerados. Pesos grandes, datasets e artefatos de execucao nao devem ser versionados.

## Requisitos

- Python 3.10+
- Node.js 18+
- npm
- Opcional: GPU NVIDIA com instalacao PyTorch/CUDA compativel

## Instalacao

### Backend

Na raiz do projeto:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Em Linux/macOS, ative o ambiente com:

```bash
source .venv/bin/activate
```

### Frontend

```bash
cd frontend
npm install
```

## Execucao

### API

Na raiz do projeto, com o ambiente Python ativado:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

A documentacao interativa fica em:

```text
http://localhost:8001/docs
```

### Interface Web

Em outro terminal:

```bash
cd frontend
npm run dev
```

Acesse:

```text
http://localhost:5173/
```

## Variaveis De Ambiente

Crie `frontend/.env` quando a API nao estiver no endereco padrao:

```env
VITE_API_URL=http://localhost:8001
```

Se `VITE_API_URL` nao for definida, o frontend usa `http://localhost:8001`.

## Modelos Suportados

As arquiteturas alinhadas entre frontend, inferencia e treinamento sao:

- `EfficientNetB0`
- `EfficientNetB2`
- `EfficientNetB3`
- `EfficientNetB7`
- `ResNet50`
- `MobileNetV3`

Os pesos treinados devem ficar em `models/`. A inferencia procura por nomes como:

```text
models/soybean_model_efficientnetb0.pth
models/soybean_model_efficientnet_b0.pth
models/efficientnet_b0.pth
```

Tambem e possivel selecionar uma versao especifica de pesos pelo frontend quando arquivos `.pth` treinados estao disponiveis.

## Formato Do Dataset

Cada dataset deve ficar dentro de `data/` com subpastas por classe:

```text
data/
  meu_dataset/
    Broken soybeans/
      img1.jpg
    Intact soybeans/
      img2.jpg
```

O treinamento usa `torchvision.datasets.ImageFolder`, portanto cada subpasta representa uma classe.

## Uso Da Interface

### Classificador

1. Selecione a arquitetura.
2. Se houver pesos treinados, escolha a versao.
3. Escolha imagem unica ou lote.
4. Envie os arquivos.
5. Execute a classificacao.

### Dashboard

A aba Dashboard resume os resultados do lote classificado:

- total analisado;
- confianca media;
- distribuicao por classificacao;
- distribuicao por qualidade;
- tabela estatistica.

### Treinamento

A aba Treinamento permite:

- listar modelos disponiveis;
- listar datasets encontrados em `data/`;
- configurar batch size, epocas, learning rate, paciencia e splits;
- acompanhar progresso por WebSocket;
- pausar, retomar, cancelar ou finalizar antecipadamente;
- consultar historico de treinamentos.

## API Principal

### Health Check

```http
GET /home
```

### Inferencia

```http
POST /inferencia
```

`multipart/form-data`:

- `model_name`: nome da arquitetura;
- `weight_filename`: nome opcional de um `.pth` em `models/`;
- `files`: uma ou mais imagens.

### Treinamento

Rotas com prefixo `/training`:

- `GET /models`
- `GET /datasets`
- `GET /status`
- `GET /history`
- `GET /model_versions/{model_name}`
- `POST /start`
- `POST /pause`
- `POST /resume`
- `POST /stop_early`
- `POST /cancel`
- `WS /ws`

## Pipeline De Treinamento Via Script

Tambem e possivel executar treinamentos em sequencia sem subir a API:

```bash
python -m backend.train_pipeline
```

O pipeline preserva recursos importantes do servico de treinamento:

- seed fixa para reprodutibilidade;
- `AdamW` com `weight_decay`;
- `ReduceLROnPlateau`;
- split configuravel (`random` ou `stratified`);
- sampler de treino configuravel (`shuffle` ou `weighted`);
- checkpoint configuravel por `val_loss`, `val_accuracy` ou `val_macro_f1`;
- congelamento inicial de backbone;
- gradient accumulation;
- execucao opcional ate `num_epochs` com `early_stopping: False`;
- tratamento de runtime para Windows;
- metricas de eficiencia para comparar modelos;
- relatorios individuais e resumo consolidado.

A pipeline comparativa combina automaticamente os modelos candidatos com os experimentos `baseline` e `experimental`, permitindo medir se as melhorias de balanceamento realmente compensam em qualidade e custo.

## Validacao

Comandos usados para validar a integracao:

```bash
cd frontend
npm run build
```

```bash
python -m compileall backend
```

## Observacoes

- Pacotes CUDA devem ser instalados conforme a plataforma seguindo a documentacao oficial do PyTorch.
- O frontend usa `VITE_API_URL` para chamadas REST e converte o mesmo host para WebSocket no treinamento.
- O repositorio mantem scripts legados em `src/models/` e processamento auxiliar em `backend/network/processador/` para referencia e experimentacao.
