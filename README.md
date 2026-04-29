# 🌱 SoyNet - Sistema de Classificação de Grãos de Soja

Sistema web completo para classificação automatizada de grãos de soja utilizando modelos de inteligência artificial. Combina uma API robusta em **FastAPI** (Python) com uma interface moderna em **React/TypeScript**.

## 📋 Visão Geral

O **SoyNet** utiliza redes neurais profundas (CNNs e EfficientNet) para classificar grãos de soja em diferentes categorias de qualidade:

- ✅ **Soja Integral** - Alta qualidade
- ⚠️ **Soja com Defeitos Leves** - Qualidade aceitável
- ❌ **Soja com Defeitos Moderados** - Qualidade comprometida
- 🚫 **Soja Quebrada/Danificada** - Rejeição

### Funcionalidades Principais

- 📸 Upload de imagem única ou lote de imagens
- 🤖 Inferência com múltiplos modelos (EfficientNet-B0, CNN customizado)
- 📊 Resultados detalhados com confiança percentual
- 🎯 Detecção de defeitos específicos
- ⚡ Processamento rápido com GPU (CUDA) ou CPU
- 🔄 API RESTful com CORS habilitado

---

## 🏗️ Arquitetura do Projeto

```
SoyNet/
├── 📁 backend/                          # API FastAPI (Python)
│   ├── main.py                          # Inicialização FastAPI e endpoints
│   ├── schemas.py                       # Modelos Pydantic (contrato API)
│   ├── requirements.txt                 # Dependências Python
│   ├── routes/
│   │   └── inference_routes.py          # Rota de inferência (referência)
│   ├── services/
│   │   └── inference_service.py         # Lógica de negócio de inferência
│   ├── network/
│   │   └── models/                      # Modelos e arquiteturas
│   └── README.md                        # Documentação backend
│
├── 📁 frontend/                         # Interface React/Vite (TypeScript)
│   ├── src/
│   │   ├── app/
│   │   │   ├── App.tsx                  # Componente principal
│   │   │   └── components/
│   │   │       ├── ModelSelector.tsx    # Seletor de modelo
│   │   │       ├── InputModeSelector.tsx# Seletor modo (único/lote)
│   │   │       ├── FileUploader.tsx     # Upload de arquivos
│   │   │       ├── FileSelector.tsx     # Gerenciador de files
│   │   │       ├── ClassificationResults.tsx # Visualização resultados
│   │   │       ├── ModelPipeline.tsx    # Pipeline de classificação
│   │   │       └── ui/                  # Componentes Radix UI
│   │   ├── styles/
│   │   │   ├── index.css
│   │   │   ├── tailwind.css
│   │   │   └── theme.css
│   │   └── main.tsx                     # Entry point React
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── README.md                        # Documentação frontend
│
├── 📁 src/                              # Scripts ML/Training (Python)
│   ├── models/
│   │   ├── model_efficientNet.py        # Treino EfficientNet-B7
│   │   └── inference_efficientNet.py    # Inferência standalone EfficientNet
│   └── visualization/
│       └── graphic.py                   # Gráficos de métricas
│
├── 📁 processa_soja/                    # Pré-processamento de imagens
│   └── processador.py                   # Script de preparação de dados
│
├── 📁 notebooks/                        # Experimentação e análise
│   └── segmentation_images.py           # Segmentação de grãos
│
├── 📁 data/                             # Dataset
│   └── processed/                       # Imagens processadas
│
├── 📁 env/                              # Ambiente virtual Python
│
├── requirements.txt                     # Deps Python (raiz)
├── .vscode/                             # Configurações VS Code
├── README.md                            # Este arquivo
└── .gitignore
```

---

## 🚀 Início Rápido

### Pré-requisitos

Antes de começar, instale:

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **pip** (gerenciador Python)
- **npm** ou **pnpm** (gerenciador Node)
- *(Opcional)* **Git** para clonar o repositório

Verifique as instalações:

```bash
python --version    # Deve ser 3.10+
node --version      # Deve ser 18+
npm --version       # Deve ser 8+
```

---

## 💻 Instalação Completa

### 1️⃣ Clonar/Preparar o Projeto

```bash
# Se usar Git
git clone <repositorio-url>
cd SoyNet

# Ou acesse o diretório do projeto
cd SoyNet
```

### 2️⃣ Configurar Backend (FastAPI)

```bash
# Criar ambiente virtual Python
python -m venv env

# Ativar ambiente virtual
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3️⃣ Configurar Frontend (React)

```bash
# Acessar diretório frontend
cd frontend

# Instalar dependências
npm install
# OU com pnpm:
pnpm install

# Voltar ao diretório raiz
cd ..
```

---

## ⚙️ Configuração de Modelos

### Baixar Pesos dos Modelos

Os arquivos de pesos dos modelos (`.pth`) devem ser colocados em `models/`:

```
backend/network/models/
└── efficientnet.pth                    # EfficientNet-B0

models/
└── soybean_model_efficientnet_b7.pth  # EfficientNet-B7 (opcional neste caminho)
```

Se os modelos não existirem, treine-os executando:

```bash
# EfficientNet-B7
python src/models/model_efficientNet.py
```

---

## ▶️ Executando o Projeto

### Terminal 1: Backend (FastAPI)

```bash
# Com ambiente virtual ativado
uvicorn backend.main:app --reload --port 8001
```

Saída esperada:
```
INFO:     Uvicorn running on http://127.0.0.1:8001
INFO:     Application startup complete
```

Você pode acessar a documentação interativa em: **http://localhost:8001/docs**

### Terminal 2: Frontend (React)

```bash
cd frontend

# Com npm
npm run dev

# OU com pnpm
pnpm dev
```

Saída esperada:
```
VITE v6.3.5  ready in 245 ms

➜  Local:   http://localhost:5173/
➜  press h + enter to show help
```

Acesse a interface em: **http://localhost:5173**

---

## 📖 Como Usar o Sistema

### Passo 1: Abrir a Interface

Acesse [http://localhost:5173](http://localhost:5173) no navegador.

### Passo 2: Selecionar Modelo

Na seção **"Configuração da Análise"**, escolha o modelo de IA:

- **EfficientNet-B0** - Mais veloz
- **EfficientNet-B7** - Maior precisão

### Passo 3: Escolher Modo de Entrada

Selecione como deseja processar imagens:

- **Imagem Única**: Analisa uma foto por vez
- **Pasta com Múltiplas**: Processa lote de imagens

### Passo 4: Fazer Upload

Clique em **"Selecionar Imagem"** ou **"Selecionar Pasta"** e escolha os arquivos.

Formatos suportados: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Passo 5: Iniciar Classificação

Clique no botão **"Iniciar Classificação"**.

O sistema irá processar e retornar resultados em segundos.

### Passo 6: Visualizar Resultados

Para cada imagem, você verá:

| Campo | Descrição |
|-------|-----------|
| **Preview** | Miniatura da imagem processada |
| **Classificação** | Tipo e qualidade do grão |
| **Confiança** | Porcentagem de certeza (0-100%) |
| **Categoria** | Tipo de grão (Tipo 1, 2, 3, 4) |
| **Qualidade** | Excelente / Boa / Regular / Ruim |
| **Defeitos** | Lista de problemas detectados |

---

## 🎛️ Treinamento e Fine-Tuning Funcional

A principal novidade da arquitetura de produção do SoyNet é o pipeline de treinamento guiado integralmente via frontend React acoplado dinamicamente com MLOps nas dependências internas PyTorch.

### Treinar e Criar Novas Redes Customizadas
1. Na Dashboard de Treinamento, escolha a baseline a ser aprimorada (`MobileNet`, `ResNet`, `EfficientNet`).
2. Defina Hiperparâmetros como **Batch Size**, **Learning Rate** e proporção (Splits T/V) entre as frações que avaliarão contra falsos positivos.
3. Tratamento Dinâmico: A injeção balanceadora de classe ($1/\sqrt{n}$) mitiga instantaneamente gargalos de datasets assimétricos desiguais entre si.
4. Feedback em tempo de execução: Um Loader interativo calcula simultaneamente o erro (Loss Difference) separando a variação gráfica.

### Controle Tático por Threading e Hardware
Como treinamentos profundos exigem alocação integral da CPU/GPU, os recursos foram isolados:
- ⏸️ **Pausar / Retomar**: A qualquer momento pare o disparo de Tensores caso haja pico externo do hardware e retriangule para Retomar posteriormente, sem fechar a sessão WebSocket.
- 💾 **Finalizar & Salvar (Stop Early)**: Precisa queimar etapas e parar de esperar o limitador atingir o teto de *paciência*? Você intercepta a thread e empurra imediatamente a IA a cruzar sua métrica da "época congelada", montando avaliativos em base no que ela compreendeu até aquele exato milissegundo.

### Histórico Científico de Resultados
Resultados da validação de teste do Fine-Tuning jamais se perdem e permanecem alvos num Banco `JSON` servidos de ponta-a-ponta:
- Gráficos Lineares **Curva ROC e AUC %**.
- **Matriz de Confusão em Heatmap**.
- Cálculo analítico multiclasse para Métricas precisas de Regressão Logística (`F1-Score, Precision, Recall e Support`).

---

## 🔌 Estrutura da API

### Endpoint Principal: POST /inferencia

**URL:** `http://localhost:8001/inferencia`

**Tipo:** POST (multipart/form-data)

**Parâmetros:**

```
model_name: string   # Nome do modelo ("EfficientNetB0" ou "EfficientNetB7")
files: file[]        # Uma ou mais imagens
```

**Resposta Sucesso (200):**

```json
[
  {
    "filename": "imagem1.jpg",
    "classification": "Soja Integral de Alta Qualidade",
    "model_used": "EfficientNetB0",
    "confidence": 95,
    "details": {
      "category": "Grão Tipo 1",
      "quality": "Excelente",
      "defects": []
    }
  },
  {
    "filename": "imagem2.jpg",
    "classification": "Soja com Defeitos Leves",
    "model_used": "EfficientNetB0",
    "confidence": 87,
    "details": {
      "category": "Grão Tipo 2",
      "quality": "Boa",
      "defects": ["Manchas leves"]
    }
  }
]
```

**Exemplos de Uso:**

#### Com cURL (Windows PowerShell):

```powershell
$filePath = "C:\caminho\imagem.jpg"
curl -Method POST `
  -Uri "http://127.0.0.1:8001/inferencia" `
  -Form @{ 
    model_name = "EfficientNetB0"
    files = Get-Item $filePath 
  }
```

#### Com Python:

```python
import requests

url = "http://127.0.0.1:8001/inferencia"

with open("imagem.jpg", "rb") as f:
    files = {"files": f}
    data = {"model_name": "EfficientNetB0"}
    
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

#### Com JavaScript/Fetch:

```javascript
const formData = new FormData();
formData.append("model_name", "EfficientNetB0");
formData.append("files", fileInput.files[0]);

const response = await fetch("http://localhost:8001/inferencia", {
  method: "POST",
  body: formData
});

const results = await response.json();
console.log(results);
```

### Endpoint de Saúde: GET /home

**URL:** `http://localhost:8001/home`

**Resposta:**

```json
{
  "message": "API de Inferência de Imagens está rodando!"
}
```

---

## 📊 Arquitetura dos Modelos

### Modelo 1: EfficientNet-B0

**Características:**
- Transfer learning com ImageNet pre-treinado
- Maior precisão (93-96%)
- Mais rápido que CNNs tradicionais
- Entrada: 224x224x3 (RGB)
- Saída: 5 classes

**Pesos:** `backend/network/models/efficientnet.pth`

```python
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, num_classes)
```

### Modelo 2: EfficientNet-B7

**Características:**
- Maior precisão (com custo computacional mais alto)
- Entrada: 600x600x3 (RGB)
- Saída: 5 classes

**Arquivos:** `src/models/model_efficientNet.py` e `src/models/inference_efficientNet.py`

---

## 🔧 Desenvolvimento

### Treinar Novos Modelos

```bash
# EfficientNet
python src/models/model_efficientNet.py

# Com MLflow para rastreamento
# (já integrado nos scripts)
```

### Testar Inferência Standalone

```bash
python src/models/inference_efficientNet.py
```

### Estrutura de Dados do Dataset

```
data/
└── processed/
  ├── Broken soybeans/           # Grãos quebrados
  ├── Immature soybeans/         # Grãos imaturos
  ├── Intact soybeans/           # Grãos íntegros
  ├── Skin-damaged soybeans/     # Danos na pele
  └── Spotted soybeans/          # Grãos manchados
```

### Pré-processamento de Imagens

```bash
python processa_soja/processador.py
```

Este script:
1. Converte para CMYK
2. Cria máscara binária
3. Separa grãos individuais
4. Remove outliers via SSIM

---

## 🐛 Solução de Problemas

### ❌ Erro: "ModuleNotFoundError: No module named 'backend'"

**Solução:**
- Garanta que está no diretório raiz do projeto
- Reative o ambiente virtual:
  ```bash
  env\Scripts\activate  # Windows
  source env/bin/activate  # Linux/Mac
  ```

### ❌ Erro: "Connection refused" ao conectar Frontend-Backend

**Solução:**
- Verifique se ambos os servidores estão rodando
- Backend deve estar em `http://127.0.0.1:8001`
- Frontend enviará requisições para esse endpoint
- Verifique se CORS está habilitado

### ❌ Erro: "Arquivo de pesos não encontrado (.pth)"

**Solução:**
- Coloque os arquivos `.pth` em `models/`
- Ou treine os modelos:
  ```bash
  python src/models/model_efficientNet.py
  ```

### ❌ CUDA não encontrado (apenas aviso, pode usar CPU)

**Mensagem:**
```
RuntimeError: No CUDA runtime found
```

**É normal!** O sistema irá usar CPU automaticamente. Para GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ Porta 5173 ou 8001 já em uso

**Solução:**
```bash
# Frontend em porta diferente
cd frontend
npm run dev -- --port 3000

# Backend em porta diferente
uvicorn backend.main:app --reload --port 8002
```

---

## 📝 Variáveis de Ambiente

Crie um arquivo `.env` na raiz (opcional):

```
# Backend
BACKEND_PORT=8001
MODEL_PATH=./models

# Frontend
VITE_API_URL=http://127.0.0.1:8001
```

---

## 🎯 Casos de Uso

### Uso Comercial
- Triagem automática de grãos em silos
- Controle de qualidade em fábricas
- Certificação de exportação

### Uso Acadêmico
- Pesquisa em visão computacional
- Benchmark de modelos CNN
- Dataset de treinamento

### Uso Agrícola
- Monitoramento de colheita
- Análise pós-colheita
- Rastreabilidade de lotes

---

## 📚 Referências e Documentação

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [PyTorch Docs](https://pytorch.org/)
- [Radix UI Components](https://www.radix-ui.com/)
- [shadcn/ui](https://ui.shadcn.com/)

---

## 📄 Licença

Este projeto contém dados e modelos proprietários. Uso restrito conforme termos definidos.

---

## 🤝 Suporte

Se encontrar problemas:

1. Verifique a seção **Solução de Problemas**
2. Verifique os logs do backend/frontend
3. Leia as documentações em `backend/README.md` e `frontend/README.md`
4. Abra uma issue no repositório

---

## 📋 Checklist de Instalação

```
✅ Python 3.10+ instalado
✅ Node.js 18+ instalado
✅ requirements.txt instalado (pip install -r requirements.txt)
✅ npm install executado em /frontend
✅ Backend rodando em http://127.0.0.1:8001
✅ Frontend rodando em http://localhost:5173
✅ Modelos .pth em /models
✅ Interface acessível e respondendo
```

---

**Desenvolvido para classificação automatizada de grãos de soja** 🌱

Versão 1.0 | Última atualização: 2025-02-08
