# Relatorio Comparativo: Melhor Modelo Atual vs YOLO26n e YOLO26s

Gerado em: 2026-05-07

Este relatorio compara o melhor resultado ja existente nos modelos tradicionais do projeto com os dois treinamentos YOLO disponiveis em `models/`.

## Artefatos analisados

| Modelo | Relatorio | Pesos |
|---|---|---|
| MobileNetV3 | `models/soybean_model_mobilenetv3_20260409_123603.txt` | `models/soybean_model_mobilenetv3_20260409_123603.pth` |
| YOLO26n-cls | `models/soybean_model_yolo26n-cls_20260506_233251.txt` | `models/yolo_runs/soybean_yolo26n-cls_20260506_221134/weights/best.pt` |
| YOLO26s-cls | `models/soybean_model_yolo26s-cls_20260507_011219.txt` | `models/yolo_runs/soybean_yolo26s-cls_20260506_233344/weights/best.pt` |

O melhor modelo tradicional encontrado foi o `MobileNetV3`, com `94.08%` de acuracia. Ele superou os demais relatorios tradicionais salvos em `models/`, incluindo `EfficientNetB2` com `93.85%`, `ResNet50` com `93.79%` e `EfficientNetB0` com `93.64%`.

## Resumo executivo

| Modelo | Accuracy test | Macro F1 | Weighted F1 | Tempo de treino | Tempo medio de inferencia |
|---|---:|---:|---:|---:|---:|
| MobileNetV3 | 94.08% | 85.71% | 94.22% | 2843.1s | n/d |
| YOLO26n-cls | 93.28% | 82.41% | 93.21% | 4828.27s | 2.877 ms |
| YOLO26s-cls | 94.33% | 85.97% | 94.29% | 5869.02s | 3.018 ms |

Conclusao principal: o `YOLO26s-cls` apresentou o melhor resultado geral entre os tres modelos comparados. Ele ficou `+0.25` ponto percentual acima do `MobileNetV3` em acuracia de teste e `+0.07` ponto em weighted F1. A diferenca e pequena, mas consistente nas metricas agregadas.

O `YOLO26n-cls` ficou abaixo do melhor modelo tradicional. Ele e uma baseline leve e rapida para inferencia, mas perdeu em acuracia, macro F1 e weighted F1.

## Observacao sobre equivalencia dos testes

A comparacao e valida como benchmark pratico, mas nao e perfeitamente identica do ponto de vista experimental.

O pipeline tradicional usa `random_split` global sobre o dataset inteiro. O pipeline YOLO criado para este projeto gera um split estratificado por classe em `data/_yolo_cls/sem_fundo`, preservando amostras em `train`, `val` e `test` para cada classe. Por isso, o suporte total do teste ficou ligeiramente diferente:

| Modelo | Imagens no teste |
|---|---:|
| MobileNetV3 | 4844 |
| YOLO26n-cls | 4850 |
| YOLO26s-cls | 4850 |

Essa diferenca e pequena e nao muda a leitura principal, mas deve ser mencionada se os resultados forem usados em trabalho academico ou documentacao formal.

## Comparacao global

### MobileNetV3

O `MobileNetV3` era o melhor resultado antes da entrada da YOLO:

- Accuracy: `94.08%`
- Macro F1: `85.71%`
- Weighted F1: `94.22%`
- Melhor classe por F1: `RGB_JPG_MAMONAS_SF` com `100.00%`
- Pior classe por F1: `RGB_JPG_INSETOS_SF` com `55.38%`
- Tempo total de treino: `2843.1s`

Ele continua sendo muito competitivo. Tambem treinou mais rapido que as duas YOLOs neste conjunto de execucoes.

### YOLO26n-cls

O `YOLO26n-cls` teve desempenho inferior ao `MobileNetV3`:

- Accuracy test: `93.28%`
- Top1 de validacao Ultralytics: `93.51%`
- Top5: `100.00%`
- Macro F1: `82.41%`
- Weighted F1: `93.21%`
- Tempo medio de inferencia: `2.877 ms`
- Tempo de treino registrado no `results.csv`: `4828.27s`

O ponto mais fraco do `YOLO26n-cls` foi a classe `RGB_JPG_INSETOS_SF`, com F1 de apenas `45.45%`. Isso prejudica principalmente o macro F1, que da o mesmo peso para classes grandes e pequenas.

### YOLO26s-cls

O `YOLO26s-cls` foi o melhor resultado geral da comparacao:

- Accuracy test: `94.33%`
- Top1 de validacao Ultralytics: `94.46%`
- Top5: `100.00%`
- Macro F1: `85.97%`
- Weighted F1: `94.29%`
- Tempo medio de inferencia: `3.018 ms`
- Tempo de treino registrado no `results.csv`: `5869.02s`

Ele melhora substancialmente sobre o `YOLO26n-cls`, especialmente nas classes minoritarias `RGB_JPG_INSETOS_SF` e `RGB_JPG_ESVERDEADO_SF`. O custo e maior tempo de treino e inferencia ligeiramente mais lenta.

## F1 por classe

| Classe | MobileNetV3 F1 | YOLO26n F1 | YOLO26s F1 | YOLO26s vs MobileNet | YOLO26s vs YOLO26n |
|---|---:|---:|---:|---:|---:|
| RGB_JPG_CHOCHOS_SF | 84.92% | 80.52% | 83.26% | -1.66 p.p. | +2.74 p.p. |
| RGB_JPG_ESVERDEADO_SF | 69.77% | 68.67% | 75.21% | +5.44 p.p. | +6.54 p.p. |
| RGB_JPG_IMPUREZAS_SF | 99.71% | 99.60% | 99.71% | +0.00 p.p. | +0.11 p.p. |
| RGB_JPG_INSETOS_SF | 55.38% | 45.45% | 61.90% | +6.52 p.p. | +16.45 p.p. |
| RGB_JPG_MAMONAS_SF | 100.00% | 97.35% | 99.12% | -0.88 p.p. | +1.77 p.p. |
| RGB_JPG_Normais_SF | 95.34% | 95.22% | 95.69% | +0.35 p.p. | +0.47 p.p. |
| RGB_JPG_PURPURAS_SF | 88.00% | 81.93% | 80.00% | -8.00 p.p. | -1.93 p.p. |
| RGB_JPG_QUEBRADOS_SF | 92.57% | 90.58% | 92.83% | +0.26 p.p. | +2.25 p.p. |

## Principais leituras por classe

`RGB_JPG_IMPUREZAS_SF` esta praticamente resolvida pelos tres modelos. Todos ficam em torno de `99.6%` a `99.71%` de F1.

`RGB_JPG_Normais_SF` tambem e estavel. O `YOLO26s-cls` teve o melhor F1 da classe, com `95.69%`, contra `95.34%` do `MobileNetV3`.

`RGB_JPG_INSETOS_SF` e uma classe critica por ter pouco suporte no teste (`24` imagens nos relatorios YOLO). O `YOLO26s-cls` foi o melhor nessa classe, com `61.90%` de F1. Isso representa ganho relevante sobre o `MobileNetV3` (`55.38%`) e principalmente sobre o `YOLO26n-cls` (`45.45%`).

`RGB_JPG_PURPURAS_SF` e o principal ponto fraco do `YOLO26s-cls`. O `MobileNetV3` teve `88.00%` de F1, enquanto o `YOLO26s-cls` caiu para `80.00%`. Se essa classe tiver peso operacional alto, esse ponto precisa ser investigado antes de trocar o modelo em producao.

`RGB_JPG_ESVERDEADO_SF` melhorou bastante com `YOLO26s-cls`: `75.21%` de F1 contra `69.77%` do `MobileNetV3`.

## Matriz de confusao: pontos relevantes

No `YOLO26s-cls`, a classe `RGB_JPG_Normais_SF` teve `1964` acertos em `2024` exemplos de teste, com a maior parte dos erros indo para `RGB_JPG_CHOCHOS_SF` (`46`) e `RGB_JPG_QUEBRADOS_SF` (`4`).

Na classe `RGB_JPG_CHOCHOS_SF`, o `YOLO26s-cls` acertou `455` de `542`. Os principais desvios foram para `RGB_JPG_Normais_SF` (`53`) e `RGB_JPG_ESVERDEADO_SF` (`17`).

Na classe `RGB_JPG_QUEBRADOS_SF`, o `YOLO26s-cls` acertou `583` de `659`. O principal erro foi confundir com `RGB_JPG_Normais_SF` (`46`) e `RGB_JPG_CHOCHOS_SF` (`22`).

Essas confusoes indicam que parte do erro esta concentrada entre classes visualmente proximas ou classes que podem depender de detalhes pequenos da textura e formato do grao.

## Custo computacional

O `MobileNetV3` foi mais eficiente no tempo total de treino:

| Modelo | Tempo de treino | Diferenca contra MobileNetV3 |
|---|---:|---:|
| MobileNetV3 | 2843.1s | referencia |
| YOLO26n-cls | 4828.27s | +1985.17s |
| YOLO26s-cls | 5869.02s | +3025.92s |

O `YOLO26s-cls` entregou o melhor resultado, mas treinou cerca de `2.06x` mais tempo que o `MobileNetV3` nesta execucao.

Para inferencia, os relatorios YOLO registraram:

| Modelo | Tempo medio por imagem |
|---|---:|
| YOLO26n-cls | 2.877 ms |
| YOLO26s-cls | 3.018 ms |

O aumento de `YOLO26n` para `YOLO26s` e pequeno em inferencia: aproximadamente `0.141 ms` por imagem. Como o ganho de acuracia foi de `+1.05` p.p. e o ganho de macro F1 foi de `+3.56` p.p., o `YOLO26s-cls` parece justificar o custo adicional sobre o `YOLO26n-cls`.

## Recomendacao

O melhor candidato atual para continuar os experimentos e o `YOLO26s-cls`.

Ele deve ser tratado como novo melhor baseline porque:

- teve a maior acuracia geral: `94.33%`;
- teve o maior macro F1: `85.97%`;
- teve o maior weighted F1: `94.29%`;
- melhorou classes minoritarias importantes, especialmente `RGB_JPG_INSETOS_SF` e `RGB_JPG_ESVERDEADO_SF`;
- manteve inferencia rapida, com `3.018 ms` por imagem.

Ainda assim, o `MobileNetV3` nao deve ser descartado. Ele continua muito forte, treinou mais rapido e foi claramente melhor em `RGB_JPG_PURPURAS_SF`. Se o objetivo for menor custo de treinamento ou estabilidade com o pipeline existente, ele segue como alternativa solida.

## Proximos passos sugeridos

1. Rodar uma segunda repeticao do `YOLO26s-cls` com outra seed para verificar estabilidade.
2. Fazer uma comparacao usando exatamente o mesmo split de teste para todos os modelos.
3. Investigar imagens erradas de `RGB_JPG_PURPURAS_SF`, onde o `MobileNetV3` ainda e superior.
4. Avaliar `YOLO26m-cls` apenas se houver necessidade real de tentar ganhar mais acuracia, pois o custo de treino deve subir.
5. Adicionar tempo medio de inferencia tambem aos modelos PyTorch para comparar latencia em igualdade.
