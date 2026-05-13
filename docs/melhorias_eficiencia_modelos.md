# Melhorias Para Aumentar A Eficiencia Dos Modelos

Este relatorio resume melhorias tecnicas com potencial de aumentar a eficiencia, a estabilidade e a qualidade dos modelos treinados no Soybe.

## 1. Split Estratificado

Hoje o treino usa `random_split`. Isso pode deixar algumas classes mal distribuidas em treino, validacao e teste, principalmente se o dataset for desbalanceado.

Um split estratificado por classe tende a gerar:

- avaliacao mais confiavel;
- comparacao mais justa entre arquiteturas;
- menor risco de uma classe rara ficar sub-representada no conjunto de validacao ou teste.

## 2. Salvar Melhor Modelo Por `macro F1`

Atualmente o checkpoint e salvo pelo menor `val_loss`.

Para classificacao com classes desbalanceadas, `macro F1` pode ser um criterio melhor, porque considera o desempenho medio entre classes e reduz a chance de o modelo favorecer apenas as classes mais frequentes.

Melhoria sugerida:

- manter `val_loss` registrado;
- calcular `macro_f1` na validacao;
- permitir escolher o criterio de checkpoint:
  - `val_loss`;
  - `macro_f1`;
  - `accuracy`.

## 3. `WeightedRandomSampler` No Treino

O treinamento ja usa `class_weights` no `CrossEntropyLoss`, o que ajuda a compensar desbalanceamento.

Uma melhoria complementar seria usar `WeightedRandomSampler` no `DataLoader` de treino. Assim, classes raras aparecem com mais frequencia nos batches.

Possiveis ganhos:

- melhor aprendizado de classes minoritarias;
- maior estabilidade em datasets assimetricos;
- aumento de `recall` em classes raras.

## 4. Scheduler Mais Adequado Para Fine-Tuning

O `ReduceLROnPlateau` e seguro e conservador, mas outras estrategias podem melhorar a convergencia:

- `CosineAnnealingLR`;
- `OneCycleLR`;
- warmup seguido de cosine decay.

Essas estrategias podem ajudar o modelo a explorar melhor no inicio e refinar os pesos no final do treinamento.

## 5. Ajustar Politica De Congelamento

A pipeline usa `freeze_backbone_epochs`, o que e positivo para fine-tuning.

Possiveis ajustes:

- modelos pequenos: congelar por menos epocas;
- `EfficientNetB2` e `EfficientNetB3`: manter 2 epocas como ponto inicial;
- `EfficientNetB7`: testar 3 ou mais epocas;
- destravar progressivamente camadas finais antes do backbone inteiro.

Esse ajuste pode reduzir instabilidade no inicio do treino e melhorar a adaptacao dos pesos pre-treinados.

## 6. Augmentations Mais Modernas

As augmentations atuais sao uma boa base, mas podem ser expandidas com cuidado:

- `RandAugment`;
- `RandomErasing`;
- `MixUp`;
- `CutMix`.

Observacao: `MixUp` e `CutMix` devem ser testados com cautela em imagens de graos individuais, porque podem criar exemplos artificiais demais e prejudicar a interpretabilidade visual do problema.

## 7. Medir Eficiencia Alem Da Acuracia

Para escolher o melhor modelo, nao basta olhar apenas acuracia.

Metricas recomendadas:

- acuracia;
- macro F1;
- recall por classe;
- tempo total de treinamento;
- tempo de inferencia por imagem;
- tamanho do arquivo `.pth`;
- consumo de RAM/VRAM.

Na pratica, `EfficientNetB2` ou `EfficientNetB3` podem entregar melhor custo-beneficio que `EfficientNetB7`, mesmo que o B7 seja maior.

## Sobre Rodar Ate O Final Sem Early Stopping

Rodar ate `num_epochs` pode trazer ganho, principalmente quando:

- a validacao oscila;
- o scheduler reduz o learning rate depois de algumas epocas ruins;
- o modelo melhora depois do ponto em que o early stopping teria parado.

Como o treinamento salva o melhor checkpoint por validacao, o maior custo de desativar early stopping e tempo de execucao, nao necessariamente perda de modelo final.

Ainda assim, existe risco de overfitting nas ultimas epocas. Por isso, e importante manter o criterio de melhor checkpoint e analisar `val_loss`, `macro_f1` e metricas por classe.

## Recomendacao De Proxima Implementacao

A melhoria mais recomendada para a proxima etapa e implementar:

1. split estratificado;
2. criterio de melhor checkpoint configuravel por `macro_f1`.

Essas duas mudancas devem melhorar a confiabilidade da comparacao entre arquiteturas e beneficiar datasets desbalanceados.

## Parametros Implementados Na Branch De Eficiencia

As recomendacoes foram transformadas em parametros para permitir comparacao entre modelos:

- `split_strategy`: aceita `random` ou `stratified`;
- `checkpoint_metric`: aceita `val_loss`, `val_accuracy` ou `val_macro_f1`;
- `sampler_strategy`: aceita `shuffle` ou `weighted`;
- `early_stopping`: permite rodar ate `num_epochs` quando definido como `False`.

A pipeline comparativa usa:

```python
"split_strategy": "stratified",
"checkpoint_metric": "val_macro_f1",
"sampler_strategy": "weighted",
"early_stopping": False
```

Os resultados tambem passam a registrar metricas de eficiencia:

- imagens de treino processadas por segundo;
- imagens de teste avaliadas por segundo;
- numero de parametros;
- parametros treinaveis;
- tamanho do checkpoint em MB.
