# Alterações No Sistema De Treinamento

Este documento descreve as principais alterações feitas no projeto para tornar o processo de treinamento executável no Windows com GPU NVIDIA, mais estável, mais observável e mais fácil de operar sem depender da API.

Os pontos abaixo estão organizados por objetivo e citam os arquivos e trechos relevantes para facilitar auditoria e manutenção.

## 1. Ajuste Das Dependências Para Windows

### O que foi alterado
O arquivo [`requirements.txt`](/c:/Users/eduar/Desktop/Soybe-main/requirements.txt) foi simplificado para remover dependências CUDA empacotadas apenas para Linux, mantendo `torch` e `torchvision` como dependências principais instaláveis no Windows.

### Onde está
- [`requirements.txt:45`](/c:/Users/eduar/Desktop/Soybe-main/requirements.txt#L45)
- [`requirements.txt:47`](/c:/Users/eduar/Desktop/Soybe-main/requirements.txt#L47)

### Por que foi alterado
O arquivo anterior continha pacotes `nvidia-*` e `triton` que não possuíam wheels compatíveis com `win_amd64`. Isso impedia a instalação do ambiente no Windows via `uv`.

### Resultado esperado
- Instalação funcional no Windows.
- Possibilidade de instalar depois a build CUDA correta do PyTorch para usar GPU.

## 2. Tratamento Específico Para Runtime No Windows

### O que foi alterado
Foi adicionada lógica de runtime para detectar Windows e limitar `num_workers` do `DataLoader` a `0`.

### Onde está
- [`backend/services/training_service.py:83`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L83)
- [`backend/services/training_service.py:99`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L99)

### Por que foi alterado
No Windows, `DataLoader` com múltiplos workers usa `spawn`, o que faz subprocessos reimportarem módulos pesados como `torch`, `pandas` e `sklearn`. Isso causava:
- `MemoryError`
- `WinError 1455`
- pressão excessiva em RAM e pagefile

### Resultado esperado
- Treino mais estável no Windows.
- Menos falhas relacionadas a multiprocessing.
- Menor risco de travamento ao usar GPU.

## 3. Import Tardio Das Métricas De Avaliação

### O que foi alterado
O import de `classification_report`, `confusion_matrix`, `roc_curve` e `auc` foi movido para a etapa final de avaliação do treino.

### Onde está
- [`backend/services/training_service.py:648`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L648)
- [`backend/services/training_service.py:653`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L653)

### Por que foi alterado
Antes, esses imports ocorriam no carregamento do módulo. No Windows isso agravava o custo de `spawn` dos subprocessos do `DataLoader`.

### Resultado esperado
- Menor consumo de memória no início do treino.
- Menor custo de importação em processos auxiliares.

## 4. Execução De Treinamento Sem API

### O que foi alterado
Foi adicionada uma API interna síncrona `run_blocking` ao gerenciador de treinamento para permitir execução via script, sem thread auxiliar nem WebSocket.

### Onde está
- [`backend/services/training_service.py:308`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L308)

### Por que foi alterado
O fluxo original exigia iniciar o backend FastAPI e enviar parâmetros por endpoint. Isso dificultava experimentos em lote e comparação entre modelos.

### Resultado esperado
- Execução de pipelines de treinamento diretamente pela linha de comando.
- Reuso da mesma lógica central de treino da API.

## 5. Criação De Pipeline De Treinamento Via Script

### O que foi alterado
Foi criado um script dedicado de pipeline para rodar múltiplos treinamentos em sequência.

### Onde está
- [`backend/train_pipeline.py:1`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L1)
- [`backend/train_pipeline.py:28`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L28)
- [`backend/train_pipeline.py:400`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L400)

### Por que foi alterado
O objetivo foi permitir:
- definir vários jobs em um único arquivo
- rodar os modelos um após o outro
- comparar resultados de forma padronizada
- evitar depender do frontend ou da API para cada execução

### Resultado esperado
- Fluxo operacional mais simples.
- Melhor repetibilidade dos experimentos.

## 6. Progresso De Treinamento Mais Rico No Terminal

### O que foi alterado
O pipeline passou a imprimir:
- status
- progresso por batch
- fechamento por época
- resultado final

### Onde está
- [`backend/train_pipeline.py:165`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L165)

### Por que foi alterado
Na API original, o usuário via basicamente o status do job. Para treino local, isso era insuficiente para acompanhar evolução, detectar instabilidade ou estimar tempo restante.

### Resultado esperado
- Visibilidade melhor do treino em tempo real.
- Diagnóstico mais fácil de problemas de convergência.

## 7. Relatórios `.txt` Por Treinamento

### O que foi alterado
Cada treinamento bem-sucedido passou a gerar um relatório `.txt` ao lado do arquivo `.pth`.

### Onde está
- [`backend/train_pipeline.py:229`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L229)

### Por que foi alterado
Era necessário preservar de forma legível:
- configuração usada
- métricas finais
- runtime do treino
- histórico por época
- matriz de confusão
- AUC por classe

### Resultado esperado
- Rastreabilidade por modelo treinado.
- Comparação posterior sem depender apenas do console.

## 8. Relatórios De Erro Por Job

### O que foi alterado
Quando um job falha, agora é gerado um arquivo `*_error.txt` com a configuração e a mensagem do erro.

### Onde está
- [`backend/train_pipeline.py:324`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L324)
- [`backend/train_pipeline.py:330`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L330)

### Por que foi alterado
Sem isso, falhas intermediárias se perdiam no terminal e ficava difícil auditar por que um treinamento específico falhou.

### Resultado esperado
- Diagnóstico posterior de falhas.
- Histórico completo também para execuções malsucedidas.

## 9. Continuação Da Pipeline Mesmo Em Caso De Erro

### O que foi alterado
O pipeline foi ajustado para continuar para o próximo job mesmo que um job intermediário falhe.

### Onde está
- [`backend/train_pipeline.py:411`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L411)
- [`backend/train_pipeline.py:433`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L433)

### Por que foi alterado
Em pipelines comparativos, abortar tudo na primeira falha desperdiça tempo e impede obter resultados dos outros modelos.

### Resultado esperado
- Melhor aproveitamento de tempo de máquina.
- Maior tolerância a falhas pontuais.

## 10. Resumo Consolidado Da Pipeline

### O que foi alterado
Ao final da execução, o pipeline gera um arquivo `pipeline_summary_<timestamp>.txt` com os resultados resumidos de todos os jobs.

### Onde está
- [`backend/train_pipeline.py:367`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L367)
- [`backend/train_pipeline.py:447`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L447)

### Por que foi alterado
Havia relatórios individuais, mas faltava uma visão consolidada para responder rapidamente:
- quais jobs passaram
- quais falharam
- qual modelo foi mais rápido
- qual teve melhor métrica

### Resultado esperado
- Comparação mais prática entre modelos.
- Base para tomada de decisão sobre qual arquitetura continuar usando.

## 11. Seed Fixa Para Reprodutibilidade

### O que foi alterado
Foi adicionada uma função de seed e o `random_split` passou a usar um `torch.Generator` com seed fixa.

### Onde está
- [`backend/services/training_service.py:116`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L116)
- [`backend/services/training_service.py:376`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L376)
- [`backend/services/training_service.py:422`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L422)

### Por que foi alterado
Sem seed fixa, cada treino usa splits ligeiramente diferentes, o que enfraquece a comparação entre arquiteturas e hiperparâmetros.

### Resultado esperado
- Comparações mais justas entre modelos.
- Melhor repetibilidade de resultados.

## 12. Otimizador Alterado Para `AdamW`

### O que foi alterado
Foi criado um construtor de otimizador que usa `AdamW` com `weight_decay`.

### Onde está
- [`backend/services/training_service.py:155`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L155)
- [`backend/services/training_service.py:475`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L475)

### Por que foi alterado
Para fine-tuning em modelos pré-treinados, `AdamW` tende a ser uma escolha mais robusta que `Adam`, principalmente pela forma correta de aplicar regularização.

### Resultado esperado
- Melhor generalização.
- Treino mais estável.

## 13. Adição De Scheduler De Learning Rate

### O que foi alterado
Foi adicionado `ReduceLROnPlateau` para reduzir o learning rate automaticamente quando `val_loss` para de melhorar.

### Onde está
- [`backend/services/training_service.py:170`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L170)
- [`backend/services/training_service.py:476`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L476)
- [`backend/services/training_service.py:605`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L605)

### Por que foi alterado
Treinos em fine-tuning se beneficiam de uma redução controlada de LR para sair da fase grossa de aprendizado e entrar em ajuste fino sem instabilidade.

### Resultado esperado
- Convergência mais suave.
- Menor chance de `val_loss` oscilar sem controle.

## 14. Treino Em Duas Fases

### O que foi alterado
Foi implementada a capacidade de congelar o backbone no início e depois destravá-lo para fine-tuning completo.

### Onde está
- [`backend/services/training_service.py:143`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L143)
- [`backend/services/training_service.py:472`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L472)
- [`backend/services/training_service.py:519`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L519)

### Por que foi alterado
Esse modo permite:
- primeiro adaptar a cabeça classificadora
- depois ajustar o backbone inteiro com mais segurança

Isso é especialmente útil em fine-tuning com datasets específicos.

### Resultado esperado
- Menor risco de destruir rapidamente os pesos pré-treinados.
- Maior estabilidade no início do treino.

## 15. Gradient Accumulation Para Modelos Pesados

### O que foi alterado
Foi adicionada lógica de `accumulation_steps`, aplicada no backward e no momento de atualizar o otimizador.

### Onde está
- [`backend/services/training_service.py:377`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L377)
- [`backend/services/training_service.py:560`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L560)
- [`backend/services/training_service.py:565`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L565)

### Por que foi alterado
Modelos como `EfficientNetB7` não cabem com batch real grande em uma RTX 3070 de 8 GB. O accumulation simula um batch efetivo maior sem exigir essa VRAM de uma vez só.

### Resultado esperado
- Treino mais estável em modelos grandes.
- Melhor uso da GPU em cenários limitados por VRAM.

## 16. Histórico Por Época Persistido No Resultado

### O que foi alterado
Cada época passou a registrar:
- loss de treino
- loss de validação
- learning rate corrente
- fase do treino
- tempo acumulado

### Onde está
- [`backend/services/training_service.py:486`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L486)
- [`backend/services/training_service.py:608`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L608)
- [`backend/services/training_service.py:745`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L745)

### Por que foi alterado
Antes só havia uma visão final do treino. Isso dificultava analisar:
- em que época houve melhora real
- quando o LR caiu
- se a fase congelada ajudou

### Resultado esperado
- Melhor capacidade de diagnóstico.
- Base para gráficos e comparações futuras.

## 17. Metadados De Runtime No Resultado Final

### O que foi alterado
O resultado final do treino passou a incluir um bloco `runtime` com informações como:
- `device`
- `num_workers`
- `pin_memory`
- `mixed_precision`
- `optimizer`
- `scheduler`
- `effective_batch_size`
- `input_size`

### Onde está
- [`backend/services/training_service.py:746`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py#L746)

### Por que foi alterado
Sem esses metadados, os relatórios finais não explicavam exatamente em que condições o modelo foi treinado.

### Resultado esperado
- Maior clareza operacional.
- Melhor auditoria dos experimentos.

## 18. Reconfiguração Da Ordem Da Pipeline

### O que foi alterado
A ordem dos modelos no pipeline foi reorganizada para executar primeiro os modelos mais rápidos e com melhor custo-benefício.

### Onde está
- [`backend/train_pipeline.py:28`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py#L28)

### Ordem atual
1. `MobileNetV3`
2. `EfficientNetB0`
3. `EfficientNetB2`
4. `ResNet50`
5. `EfficientNetB7`

### Por que foi alterado
Os resultados em `models/*.txt` mostraram que:
- `MobileNetV3` e `EfficientNetB0` são os mais rápidos
- `ResNet50` tem boa acurácia, mas é mais lenta
- `EfficientNetB7` foi muito lenta e teve pior desempenho

### Resultado esperado
- Feedback mais rápido no início da pipeline.
- Menor tempo até obter modelos úteis.

## 19. Documentação Atualizada

### O que foi alterado
O README do backend foi atualizado para documentar o novo pipeline, seus recursos e artefatos gerados.

### Onde está
- [`backend/README.md:39`](/c:/Users/eduar/Desktop/Soybe-main/backend/README.md#L39)

### Por que foi alterado
O sistema deixou de ser apenas uma API de inferência e passou a ter um fluxo de treinamento local relevante. Isso precisava ficar documentado.

### Resultado esperado
- Onboarding mais simples.
- Menor dependência de conhecimento implícito.

## Resumo Executivo

As alterações feitas no sistema atacaram quatro problemas principais:

1. Compatibilidade com Windows e estabilidade de runtime.
Isso foi resolvido principalmente com ajuste de dependências, `num_workers=0` no Windows e import tardio de métricas.

2. Operação prática de treinamentos em lote.
Isso foi resolvido com a criação de um pipeline de CLI e com a continuação automática mesmo em caso de erro.

3. Melhoria técnica do processo de treinamento.
Isso foi resolvido com `AdamW`, scheduler, seed fixa, treino em duas fases e gradient accumulation.

4. Observabilidade e rastreabilidade.
Isso foi resolvido com relatórios `.txt`, relatórios de erro, resumo consolidado e histórico por época.

## Arquivos Principais Impactados

- [`requirements.txt`](/c:/Users/eduar/Desktop/Soybe-main/requirements.txt)
- [`backend/services/training_service.py`](/c:/Users/eduar/Desktop/Soybe-main/backend/services/training_service.py)
- [`backend/train_pipeline.py`](/c:/Users/eduar/Desktop/Soybe-main/backend/train_pipeline.py)
- [`backend/README.md`](/c:/Users/eduar/Desktop/Soybe-main/backend/README.md)

