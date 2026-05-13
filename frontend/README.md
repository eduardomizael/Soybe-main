# Frontend (React + Vite)

Interface para:

- classificar imagens
- acompanhar resultados
- iniciar e monitorar treinamento

## Requisitos

- Node.js 18+

## Instalacao

```bash
cd frontend
npm install
```

## Variaveis de ambiente

Crie frontend/.env:

```bash
VITE_API_URL=http://localhost:8001
```

Se nao for definida, o app usa fallback para http://localhost:8001.

## Desenvolvimento

```bash
npm run dev
```

## Build

```bash
npm run build
```

## Fluxo principal

- Aba Classifier: seleciona modelo/versao, envia imagens e recebe predicoes.
- Aba Dashboard: visualiza agregados dos resultados.
- Aba Training: controla treino e acompanha progresso em tempo real por WebSocket.

## Notas

- As chamadas de API e WS usam VITE_API_URL (com conversao http->ws quando necessario).
