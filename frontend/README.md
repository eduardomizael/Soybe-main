# Frontend

Interface React/Vite do Soybe para classificacao de imagens, dashboard de resultados e controle de treinamento.

## Requisitos

- Node.js 18+
- npm

## Instalacao

```bash
npm install
```

## Desenvolvimento

```bash
npm run dev
```

URL padrao:

```text
http://localhost:5173/
```

## Build

```bash
npm run build
```

Os arquivos de producao sao gerados em `dist/`.

## Configuracao Da API

Crie `frontend/.env` quando a API nao estiver em `localhost:8001`:

```env
VITE_API_URL=http://localhost:8001
```

O mesmo valor e usado para chamadas REST e para montar o WebSocket de treinamento.

## Telas

### Classificador

- seleciona arquitetura;
- busca versoes treinadas em `/training/model_versions/{model_name}`;
- envia imagens para `/inferencia`;
- mostra resultados individuais.

### Dashboard

- agrega resultados do lote;
- mostra composicao por classificacao;
- mostra distribuicao por qualidade;
- exibe resumo estatistico.

### Treinamento

- lista modelos em `/training/models`;
- lista datasets em `/training/datasets`;
- inicia treinamento com `/training/start`;
- controla pausa, retomada, cancelamento e finalizacao antecipada;
- acompanha progresso via `WS /training/ws`;
- mostra historico de treinamentos.

## Modelos Exibidos

- `EfficientNetB0`
- `EfficientNetB2`
- `EfficientNetB3`
- `EfficientNetB7`
- `ResNet50`
- `MobileNetV3`

## Estrutura Relevante

```text
src/
  app/
    App.tsx
    components/
      Dashboard.tsx
      Navbar.tsx
      TrainingDashboard.tsx
      ModelSelector.tsx
      ModelPipeline.tsx
      ClassificationResults.tsx
      FileUploader.tsx
      InputModeSelector.tsx
      ui/
  styles/
```

## Observacoes

- O frontend nao usa dados simulados no fluxo principal; ele chama a API FastAPI.
- O seletor de versao usa os arquivos `.pth` encontrados em `models/` pelo backend.
- `sonner` e usado para notificacoes.
- `recharts` e usado nos dashboards e graficos de treinamento.
