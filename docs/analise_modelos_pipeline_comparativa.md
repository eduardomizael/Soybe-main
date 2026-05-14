# Analise Dos Modelos Para Pipeline Comparativa

Este relatorio resume a leitura dos arquivos `.txt` existentes em `models/` para definir quais arquiteturas valem ser executadas com duas configuracoes de treinamento:

- `baseline`;
- `experimental`.

## Configuracoes Comparadas

### baseline

```python
"split_strategy": "random",
"checkpoint_metric": "val_loss",
"sampler_strategy": "shuffle",
"early_stopping": True
```

Essa configuracao representa o comportamento tradicional e serve como linha de base neutra.

### experimental

```python
"split_strategy": "stratified",
"checkpoint_metric": "val_macro_f1",
"sampler_strategy": "weighted",
"early_stopping": False
```

Essa configuracao testa melhorias voltadas a datasets desbalanceados e comparacao por macro F1.

## Leitura Dos Relatorios Existentes

| Modelo | Melhor accuracy | Macro F1 estimado | Tempo | Observacao |
|--------|-----------------|-------------------|-------|------------|
| MobileNetV3 | 94.08% | 85.71% | 2843.1s | Melhor custo-beneficio atual. |
| EfficientNetB2 | 93.85% | 85.48% | 5945.7s | Melhor EfficientNet ja testada. |
| ResNet50 | 93.79% | 85.11% | 6576.6s | Boa qualidade, mas lenta frente a MobileNetV3/B2. |
| EfficientNetB0 | 93.64% | 83.53% | 3852.2s | Rapida e competitiva. |
| EfficientNetB7 | 31.00% | 25.64% | 18171.6s | Custo alto e resultado ruim nos relatorios atuais. |

Nao havia relatorio historico para `EfficientNetB3`, mas ela foi incluida por ser um ponto intermediario natural entre `EfficientNetB2` e `EfficientNetB7`.

## Modelos Selecionados

A pipeline comparativa padrao executa:

- `MobileNetV3`;
- `EfficientNetB0`;
- `EfficientNetB2`;
- `EfficientNetB3`.

## Modelos Mantidos Fora Da Comparacao Padrao

### ResNet50

Foi mantido fora da execucao padrao porque teve qualidade parecida com `EfficientNetB2`, mas tempo maior. Pode ser reativado em `CANDIDATE_MODELS` se houver interesse em comparar robustez de arquitetura.

### EfficientNetB7

Foi mantido fora da execucao padrao porque os relatorios indicaram baixo desempenho e tempo muito alto. Antes de reexecutar B7, seria melhor revisar batch, accumulation, input size, memoria disponivel e estabilidade do dataset.

## Resultado Esperado Da Nova Pipeline

Com quatro modelos e dois experimentos, a pipeline padrao gera oito jobs:

```text
MobileNetV3 baseline
MobileNetV3 experimental
EfficientNetB0 baseline
EfficientNetB0 experimental
EfficientNetB2 baseline
EfficientNetB2 experimental
EfficientNetB3 baseline
EfficientNetB3 experimental
```

Os relatorios gerados passam a incluir:

- nome do experimento;
- parametros de split/checkpoint/sampler;
- accuracy;
- macro F1;
- melhor metrica de checkpoint;
- tempo total;
- throughput de treino e teste;
- tamanho do checkpoint;
- numero de parametros.
