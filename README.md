# Soybe

Sistema de classificação de grãos de soja utilizando redes neurais convolucionais,
com backend em **FastAPI** e frontend em **React + Vite**.

## Estrutura do Projeto

```
Soybe/
├── backend/            # API REST (FastAPI)
│   ├── main.py         # Inicialização, CORS e registro de rotas
│   ├── schemas.py      # Contratos Pydantic
│   ├── routes/
│   │   ├── inference_routes.py
│   │   └── training_routes.py
│   └── services/
│       ├── inference_service.py
│       └── training_service.py
├── frontend/           # Interface web (React + Vite + TailwindCSS)
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx
│   │   │   └── components/
│   │   └── styles/
│   └── ...
├── data/               # Datasets locais para treinamento (não versionado)
├── models/             # Pesos treinados e histórico (não versionado)
├── requirements.txt    # Dependências Python
└── README.md
```

## Requisitos

- Python 3.10+
- Node.js 18+

## Setup Rápido

### 1) Backend

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Iniciar API:

```bash
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Crie `frontend/.env` com:

```bash
VITE_API_URL=http://localhost:8001
```

Em produção use a URL pública da API.

## Endpoints Principais

### Inferência

| Método | Rota          | Descrição                      |
|--------|---------------|--------------------------------|
| POST   | `/inferencia` | Classifica uma ou mais imagens |
| GET    | `/home`       | Health check                   |

### Treinamento (prefixo `/training`)

| Método | Rota                          | Descrição                          |
|--------|-------------------------------|------------------------------------|
| GET    | `/models`                     | Lista modelos disponíveis          |
| GET    | `/datasets`                   | Lista datasets em `data/`          |
| POST   | `/start`                      | Inicia treinamento                 |
| GET    | `/status`                     | Status do treinamento atual        |
| POST   | `/pause`                      | Pausa treinamento                  |
| POST   | `/resume`                     | Retoma treinamento                 |
| POST   | `/stop_early`                 | Interrompe com salvamento          |
| POST   | `/cancel`                     | Cancela treinamento                |
| GET    | `/history`                    | Histórico de treinos               |
| GET    | `/model_versions/{model}`     | Versões de pesos por arquitetura   |
| WS     | `/ws`                         | Progresso em tempo real            |

## Formato de Dataset

Cada dataset deve ficar dentro de `data/` com subpastas por classe:

```
data/
  meu_dataset/
    classe_a/
      img1.jpg
    classe_b/
      img2.jpg
```

## Arquiteturas Suportadas

- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B7
- ResNet-50
- MobileNet-V3

Os pesos são salvos em `models/` com nomenclatura por arquitetura e timestamp.
