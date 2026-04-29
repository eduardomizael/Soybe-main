# Sistema de Classificação de Grãos de Soja

Sistema web para classificação de grãos de soja utilizando modelos de inteligência artificial treinados.

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter instalado em sua máquina:

- **Node.js** (versão 18 ou superior) - [Download](https://nodejs.org/)
- **npm** ou **pnpm** (gerenciador de pacotes)

Para verificar se o Node.js está instalado:
```bash
node --version
npm --version
```

## 🚀 Instalação

### 1. Clone ou baixe o projeto

```bash
# Se estiver usando Git
git clone <url-do-repositorio>
cd <nome-do-projeto>
```

### 2. Instale as dependências

Escolha um dos comandos abaixo de acordo com seu gerenciador de pacotes:

```bash
# Usando npm
npm install

# OU usando pnpm (recomendado)
pnpm install
```

Este comando irá instalar todas as dependências necessárias, incluindo:
- React 18.3.1
- Vite (build tool)
- Tailwind CSS 4.1.12
- Radix UI (componentes)
- Lucide React (ícones)
- Sonner (notificações)

## 💻 Executando o Projeto

### Modo de Desenvolvimento

Para iniciar o servidor de desenvolvimento:

```bash
# Com npm
npm run dev

# Com pnpm
pnpm dev
```

O projeto estará disponível em: **http://localhost:5173**

O servidor irá recarregar automaticamente quando você fizer alterações no código.

### Build para Produção

Para criar uma versão otimizada para produção:

```bash
# Com npm
npm run build

# Com pnpm
pnpm build
```

Os arquivos otimizados serão gerados na pasta `dist/`.

## 📖 Como Usar o Sistema

### 1. Selecionar o Modelo Treinado

Na tela principal, escolha um dos modelos de IA disponíveis:
- **ResNet-50** - Precisão de 94.5%
- **EfficientNet-B3** - Precisão de 96.2% (recomendado)
- **VGG-16** - Precisão de 92.8%
- **MobileNet-V2** - Mais rápido, precisão de 91.3%

### 2. Escolher o Modo de Entrada

Selecione como você deseja enviar as imagens:
- **Imagem Única**: Analisa apenas uma foto por vez
- **Pasta com Múltiplas Imagens**: Analisa várias fotos em lote

### 3. Selecionar Imagens

Clique no botão de upload:
- Para **imagem única**: Selecione um arquivo de imagem (.jpg, .png, etc.)
- Para **pasta**: Selecione múltiplas imagens ou uma pasta completa

### 4. Iniciar Classificação

Clique no botão **"Iniciar Classificação"** para processar as imagens.

### 5. Visualizar Resultados

O sistema exibirá para cada imagem:
- **Preview da imagem**
- **Classificação** do grão (tipo e qualidade)
- **Nível de confiança** (porcentagem)
- **Categoria** do grão (Tipo 1, 2, 3 ou 4)
- **Qualidade** (Excelente, Boa, Regular ou Ruim)
- **Defeitos detectados** (se houver)

## 🔧 Estrutura do Projeto

```
/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── ui/                    # Componentes de interface
│   │   │   ├── ModelSelector.tsx      # Seletor de modelo
│   │   │   ├── InputModeSelector.tsx  # Seletor de modo
│   │   │   ├── FileUploader.tsx       # Upload de arquivos
│   │   │   └── ClassificationResults.tsx # Resultados
│   │   └── App.tsx                    # Componente principal
│   └── styles/
│       ├── index.css
│       ├── tailwind.css
│       └── theme.css
├── package.json
├── vite.config.ts
└── README.md
```

## 🔌 Integrando com API Real

Atualmente, o sistema utiliza dados simulados. Para integrar com uma API real de classificação:

1. Abra o arquivo `/src/app/App.tsx`
2. Localize a função `simulateClassification`
3. Substitua pela chamada real à sua API:

```typescript
const simulateClassification = async (file: File): Promise<ClassificationResult> => {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model', selectedModel);

  const response = await fetch('https://sua-api.com/classify', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  
  return {
    filename: file.name,
    imageUrl: URL.createObjectURL(file),
    classification: data.classification,
    confidence: data.confidence,
    details: {
      category: data.category,
      quality: data.quality,
      defects: data.defects || [],
    },
  };
};
```

## 🐛 Solução de Problemas

### Erro: "Cannot find module"
```bash
# Limpe o cache e reinstale
rm -rf node_modules package-lock.json
npm install
```

### Porta 5173 já está em uso
```bash
# O Vite automaticamente tentará a próxima porta disponível
# Ou você pode especificar uma porta diferente:
npm run dev -- --port 3000
```

### Imagens não são carregadas
- Verifique se os arquivos são imagens válidas (.jpg, .png, .jpeg)
- Certifique-se de que o navegador tem permissão para acessar arquivos locais

## 📦 Dependências Principais

| Pacote | Versão | Descrição |
|--------|--------|-----------|
| react | 18.3.1 | Framework JavaScript |
| vite | 6.3.5 | Build tool e dev server |
| tailwindcss | 4.1.12 | Framework CSS |
| lucide-react | 0.487.0 | Biblioteca de ícones |
| sonner | 2.0.3 | Sistema de notificações |

## 📝 Notas Importantes

- O sistema atualmente usa **dados simulados** para demonstração
- Para produção, é necessário integrar com um **backend real** que processe as imagens
- As classificações e scores de confiança são gerados aleatoriamente para fins de demonstração
- Recomenda-se usar **HTTPS** em produção para segurança

## 🤝 Suporte

Se encontrar problemas durante a instalação ou uso:

1. Verifique se todas as dependências foram instaladas corretamente
2. Certifique-se de estar usando Node.js versão 18 ou superior
3. Limpe o cache e reinstale as dependências se necessário
4. Verifique o console do navegador para mensagens de erro

## 📄 Licença

Este projeto é fornecido como está, sem garantias.

---

**Desenvolvido para classificação automatizada de grãos de soja** 🌱