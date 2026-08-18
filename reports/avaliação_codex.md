Parecer elaborado exclusivamente a partir do manuscrito fornecido: :codex-file-citation{path="C:\Users\eduar\Downloads\artigo1_rgb_en_versao_atual.pdf" purpose="source"}

## 1. Resumo executivo

O manuscrito apresenta uma contribuição metodológica relevante, cuidadosamente limitada a um único conjunto de dados e a uma campanha de aquisição. O contraste entre particionamento por grão e por fotografia está bem motivado, é reproduzível em princípio e permanece consistente nas arquiteturas avaliadas. A análise reconhece corretamente que “semente” combina partição e aleatoriedade de treinamento, que os blocos não são independentes e que o contraste não identifica um único mecanismo causal.

Entretanto, a Conclusão e parte da Discussão reproduzem números antigos ou incorretos justamente na decomposição de variância que sustenta RQ2: 35,8%/24,4% e crescimento “sixfold” contradizem 23,2%/41,9% e \(67.5\rightarrow515\) p.p.² apresentados nos Resultados. Isso impede aceitação da versão atual.

**As evidências sustentam as conclusões? Parcialmente.** A conclusão principal sobre o contraste entre protocolos é sustentada; a conclusão quantitativa sobre a redistribuição da variância não está internamente consistente.

## 2. Pontos fortes

- Delimitação excepcionalmente honesta da validade externa: uma campanha, um equipamento e ausência de teste externo.
- Uso correto de macro F1 como métrica principal diante do forte desbalanceamento.
- Comparação entre os protocolos no mesmo universo de 48.039 registros.
- Manifestações explícitas dos principais confundidores: tamanho de treino, amostras de teste diferentes, pesos de classe, sampler e seleção de checkpoint.
- Separação cuidadosa entre “arquitetura” e “arquitetura–receita”.
- Publicação declarada de partições, predições e matrizes, que, se os artefatos estiverem acessíveis, permite recomputar as análises centrais.
- Figuras, tabelas e diagrama de diferença crítica visualmente legíveis, sem sobreposições ou defeitos de renderização.
- Limitações de segmentação, SSIM, rotulação, sessão e implantação são declaradas, e em geral não são ignoradas na discussão.

## 3. Crítico

### C1. A Conclusão contradiz os Resultados na decomposição central de variância

**Local**

- Seção IV-H: “*Under the random per-grain protocol the same decomposition gives 23.2% to the seed and 41.9% to the architecture*”.
- Seção IV-H: “*moving to the group-disjoint protocol multiplies \(SST\) by nearly eight (67.5 to 515 p.p.²)*”.
- Discussão: “*the ratio describes shares of a total that itself grows sixfold under grouping*”.
- Conclusão: “*the seed package rises from 35.8% to 73.9% while the architecture falls from 24.4% to 6.2%*”.
- A continuação da Conclusão volta a afirmar que o total se expandiu “*sixfold*”.

**Problema**

Os valores 35,8% e 24,4% não correspondem aos resultados apresentados na Seção IV-H. Além disso,

\[
515/67.5 = 7.63,
\]

portanto o crescimento é aproximadamente 7,6 vezes, ou “nearly eightfold”, não seis vezes. A divergência está na resposta de RQ2 e na síntese final do artigo, incidindo diretamente sobre sustentação das conclusões.

**Correção**

Recomputar a seção a partir dos resultados por execução e substituir os trechos conflitantes por:

> “The protocol change also redistributes the observed variance shares. Under random per-grain partitioning, the seed package accounts for 23.2% of \(SST\) and the architecture–recipe pair for 41.9%; under photograph-disjoint partitioning, the corresponding shares are 73.9% and 6.2%. The total \(SST\) increases from 67.5 to 515 p.p.², a 7.63-fold increase, while \(SS_A\) grows from 28.3 to 32.0 p.p.². Thus, the architecture share decreases because the denominator expands, not because its absolute component disappears.”

Na Discussão, substituir:

> “a total that itself grows sixfold under grouping”

por:

> “a total that grows 7.63-fold under grouping, from 67.5 to 515 p.p.²”.

Fazer uma busca global por `35.8`, `24.4`, `sixfold`, `67.5` e `515` antes da nova submissão.

**Recurso exigido:** reanálise dos dados existentes + texto. Sem novo treinamento. Custo estimado: 30–90 minutos, desde que o script da decomposição esteja disponível.

## 4. Importantes

### I1. O Nemenyi ainda é descrito com linguagem inferencial apesar da dependência reconhecida

**Local**

- Métodos: “*its asymptotic null assumes independent blocks, which repeated partitions of one dataset are not*”.
- Resultados: “*separates only ResNet50 above EfficientNetB3*”.
- Discussão: “*the post-hoc analysis, however, separates a single pair*”.

**Problema**

O manuscrito reconhece corretamente que os dez blocos são partições sobrepostas de um único conjunto. Nesse cenário, o controle nominal de erro familiar do Nemenyi não está calibrado. “Separates” e “post-hoc analysis” ainda podem ser lidos como inferência válida, contradizendo a qualificação descritiva.

**Correção**

Na Figura 7, na Seção IV-H e na Discussão, usar:

> “Within the ten overlapping partitions observed here, the mean-rank gap exceeds the nominal Nemenyi critical difference of 2.38. Because the blocks are not independent datasets, this is a descriptive threshold crossing, not a familywise-error-controlled post-hoc inference.”

Substituir “significant difference” por “interval spanning zero” ou “no conclusive conditional difference” nos demais contrastes que usam as mesmas partições.

Caso os autores queiram preservar uma afirmação inferencial, será necessário um desenho com campanhas independentes. Repetições em partições fixas separam aleatoriedade de treinamento, mas não criam independência entre campanhas.

**Recurso exigido:** texto, para o escopo condicional atual. Novo experimento apenas para uma conclusão populacional.

### I2. “Unambiguous labels” excede o que foi demonstrado sobre o ground truth

**Local**

- Resumo: “*the benchmark uses 48,039 from the 466 with unambiguous labels*”.
- Conclusão: “*the subset of the 48,973 released whose class labels are unambiguous*”.
- Limitações: “*No intra- or inter-grader agreement was measured*” e “*the labeling unit was the batch*”.

**Problema**

O que foi resolvido é a ambiguidade administrativa de quatro fotografias armazenadas em duas classes. A correção dos rótulos restantes não foi validada por anotação independente, concordância ou julgamento individual. “Unambiguous” pode ser interpretado como ground truth validado.

**Correção**

Usar no Resumo e na Conclusão:

> “the benchmark uses 48,039 crops from 466 source photographs carrying a single dataset label, after excluding four photographs filed under conflicting classes; per-grain label accuracy and inter-rater agreement were not measured.”

Isso preserva o resultado sem exigir nova rotulação. Se os autores desejarem afirmar validade dos rótulos contra o padrão MAPA, o desenho mínimo seria uma amostra estratificada de aproximadamente 400–800 grãos, julgada independentemente por pelo menos três classificadores, com matriz de concordância e tratamento explícito das categorias limítrofes.

**Recurso exigido:** texto para a afirmação atual; novo experimento humano somente para fortalecer o ground truth.

### I3. O Resumo apresenta a etapa SSIM como detecção de outliers, mas o corpo demonstra apenas triagem

**Local**

- Resumo: “*flags outliers by structural similarity*”.
- Métodos: “*A screening module flags candidate outliers for expert review*”.
- Métodos: não foram registrados distribuição dos escores, sweep de \(\tau\), falsos negativos, concordância com especialistas ou referências originais.

**Problema**

O módulo não detecta outliers com desempenho conhecido. Ele encaminha candidatos para decisão humana, com possíveis falsos negativos entre os não sinalizados.

**Correção**

Substituir no Resumo:

> “and uses structural similarity to flag candidate outliers for expert review.”

Onde aparecer “automated dataset-construction pipeline”, preferir:

> “a semi-automated dataset-construction pipeline with automated preprocessing and expert adjudication of flagged candidates.”

**Recurso exigido:** texto.

### I4. As afirmações de disponibilidade são centrais, mas não puderam ser verificadas no material fornecido

**Local**

Seção “Data and Code Availability”, especialmente:

> “from which the analyses of Section IV [...] can be recomputed”.

**Problema**

O PDF declara DOI, release, commit, dados, manifests, predições e matrizes, mas os artefatos não fazem parte do material avaliado. Portanto, disponibilidade, conteúdo e capacidade real de recomputação permanecem indeterminados nesta rodada.

**Correção**

Antes da submissão:

- testar os dois DOIs e a URL do repositório em sessão não autenticada;
- fornecer inventário com tamanho e SHA-256 de cada pacote;
- executar, em ambiente limpo, o comando que recompõe pelo menos Tabelas 5, 10, 12, 13 e 15;
- incluir esse comando no README;
- deixar explícito que a triagem SSIM original não é reproduzível identicamente porque as referências de classe foram perdidas.

**Recurso exigido:** reanálise dos dados existentes + verificação dos artefatos. Sem novo experimento; aproximadamente 1–3 horas.

## 5. Menores

### M1. Metadado editorial inconsistente

**Local:** rodapé de todas as páginas: “*VOLUME 11, 2023*”, apesar de referências, datas de acesso e protocolo de 2026.

**Problema:** aparenta permanência de metadado do modelo IEEE.

**Correção:** usar o modelo atual de submissão e remover volume/ano predefinidos, deixando esses campos para a editora.

**Recurso exigido:** texto/formatação.

### M2. Frase gramaticalmente imprecisa na Conclusão

**Local:** “*the leading pair, ConvNeXt-Tiny, reaches 88.6% mean macro F1*”.

**Problema:** ConvNeXt-Tiny é uma arquitetura–receita, não um “pair” formado por duas candidatas.

**Correção:**

> “the architecture–recipe pair with the highest mean macro F1, ConvNeXt-Tiny under its evaluated recipe, reaches 88.6% macro F1 and 95.6% accuracy.”

**Recurso exigido:** texto.

### M3. Contagem das variantes pode ser mais explícita na disponibilidade

**Local:** “*48,973 grain instances in two variants, 97,405 files*”.

**Problema:** a frase pode sugerir correspondência completa, embora o artigo informe 48.973 arquivos preservados e 48.432 removidos.

**Correção:**

> “48,973 background-preserved files and 48,432 background-removed files, totaling 97,405 files; the variants are not in complete one-to-one correspondence.”

**Recurso exigido:** texto.

## 6. Reprodutibilidade

- ✓ Hardware: RTX 3070, 8 GB, CUDA e cuDNN declarados.
- ✓ Versões principais: Python, PyTorch, torchvision, OpenCV, scikit-image, scikit-learn e NumPy declarados.
- ✓ Hiperparâmetros: arquitetura, entrada, batch, LR, FT-LR e weight decay fornecidos.
- ✓ Seeds: dez valores informados.
- ✓ Particionamento: algoritmo, proporções, unidade de agrupamento e exceções descritos.
- ⚠ Manifests: declarados, mas não verificados nesta rodada.
- ⚠ Código: release e commit declarados, mas não verificados.
- ⚠ Dataset: DOI e inventário declarados, mas não verificados.
- ✓ Pré-processamento e augmentations: descritos.
- ✓ Critério de parada e checkpoint: descritos.
- ✓ Não determinismo total: corretamente declarado.
- ✗ Triagem SSIM original: não pode ser repetida identicamente porque as referências por classe não foram preservadas.
- ✗ Ground truth: sem anotação individual independente, concordância intra/interavaliador ou protocolo recuperável de adjudicação.
- ⚠ Recomposição das análises centrais: aparentemente possível a partir dos artefatos declarados, mas não demonstrada no PDF isolado.
- ✗ Validade externa: nenhuma campanha, instalação, câmera ou sessão externa foi avaliada; essa limitação está corretamente declarada.

## 7. Verificação numérica

### Conferido

- Tabela 2: soma do conjunto lançado = 48.973.
- Tabela 2: soma do benchmark = 48.039.
- Exclusão: \(48.973-48.039=934\) registros.
- Fotografias do benchmark: soma = 466.
- Média global: \(48.039/466=103{,}09\) grãos/fotografia, compatível com 103.
- Probabilidade de uma fotografia de 104 grãos ficar em um único subconjunto:
  \[
  0.8^{104}+2(0.1^{104})=8.34\times10^{-11}.
  \]
- Tabela 1: média das seis acurácias = 94,8617%, compatível com 94,9%.
- Tabela 5: médias arredondadas = 93,4817% e 87,745%; queda = 5,7367 p.p., compatível com 5,74.
- Gaps por classe: média dos oito valores = exatamente 5,74 p.p.
- Três classes esparsas:
  \[
  (8.87+9.84+16.75)/(8\times5.74)=77.22\%.
  \]
- Quatro classes bem amostradas: 22,52%, compatível com 23%.
- Castor bean: \(0.12/(8\times5.74)=0.26\%\), compatível com “about 0.3%”.
- Tabela 8: retenções = 94,35% e 96,55%, compatíveis com 94,3% e 96,6%.
- Tabela 11: estimativas recalculadas = 7,354%, 10,135%, 12,916% e 59,915%.
- Limite de prevalência para viés positivo:
  \[
  0.055/(0.018+0.055)=75.34\%.
  \]
- Média dos seis desvios-padrão de macro F1 = 2,973 p.p., compatível com 2,97.
- Diferença crítica:
  \[
  2.85\sqrt{42/60}=2.384,
  \]
  compatível com 2,38.
- Tabela 13: as três decomposições somam 100%.
- Tabela 14: amplitude \(89.23-86.94=2.29\) p.p.
- Valores corrigidos aproximados:
  \[
  \omega_A^2=4.0\%,\quad\omega_S^2=69.6\%,\quad
  \omega_S^2/\omega_A^2=17.53
  \]
  sob agrupamento, e razão 0,43 sob particionamento aleatório.

### Divergente

- Resultados: 23,2% semente e 41,9% arquitetura sob o protocolo aleatório.
- Conclusão: 35,8% semente e 24,4% arquitetura para a mesma condição.
- Resultados: \(SST=67.5\rightarrow515\), crescimento de 7,63 vezes.
- Discussão/Conclusão: “sixfold”.

### Não recalculável apenas pelo PDF

- CIs BCa e bootstrap por fotografia.
- Valores exatos dos testes de permutação além dos casos no limite.
- Afirmação “em todos os dez seeds e todas as arquiteturas”.
- Cinco de seis arquiteturas liderando ao menos uma vez.
- Correlações por seed e por classe.
- Decomposição por classe dos componentes de variância.
- Contagens não impressas da matriz de confusão.
- Integridade, disponibilidade e conteúdo dos artefatos externos.
- Protocolo PRISMA e contagem 172/79, pois o suplemento não foi fornecido.

Nenhum suposto erro foi inferido a partir de células rasterizadas da Figura 5.

## 8. Inglês

Somente alterações que afetam precisão:

1. Original: “*flags outliers by structural similarity*”  
   Proposta: “*flags candidate outliers for expert review using structural similarity*”.

2. Original: “*from the 466 with unambiguous labels*”  
   Proposta: “*from 466 source photographs carrying a single dataset label after exclusion of four photographs filed under conflicting classes*”.

3. Original: “*the leading pair, ConvNeXt-Tiny, reaches...*”  
   Proposta: “*the architecture–recipe pair with the highest mean macro F1, ConvNeXt-Tiny under its evaluated recipe, reaches...*”.

4. Original: “*the analysis did not detect a significant macro F1 difference*”  
   Proposta, coerente com o enquadramento descritivo: “*the paired macro-F1 contrast was inconclusive within the ten overlapping partitions, with its interval spanning zero*”.

## 9. Risco editorial

- **Alto:** números incompatíveis na Conclusão sobre RQ2.
- **Alto:** “separation/significant” aplicada a Nemenyi/Friedman com blocos reconhecidamente dependentes.
- **Alto:** disponibilidade e recomputabilidade dos artefatos, caso algum DOI, release ou arquivo esteja incompleto.
- **Médio:** “unambiguous labels” apesar de ground truth não validado.
- **Médio:** pipeline apresentado no Resumo como detecção de outliers, embora seja triagem humana não validada.
- **Médio:** título “Governs” pode ser interpretado causalmente; “Alters Reported Performance” seria mais alinhado ao contraste entre estimandos.
- **Baixo:** metadado “VOLUME 11, 2023” e pequenos ajustes terminológicos.

## 10. Parecer final

**Decisão simulada: REJECT — resubmission encouraged.**

A rejeição não decorre da ausência de teste externo, da limitação a uma campanha ou da falta de validação quantitativa da segmentação: todas são limitações declaradas e corretamente delimitadas. O bloqueio é a contradição numérica dentro da conclusão central de RQ2, agravada pela repetição do “sixfold” na Discussão e na Conclusão. A linguagem de separação estatística também precisa ser compatibilizada com a dependência entre os blocos.

Critérios IEEE Access:

- Solidez técnica: predominantemente atendida, com ressalva à interpretação nominal do Nemenyi.
- Sustentação das conclusões: não atendida integralmente devido à divergência 23,2/41,9 versus 35,8/24,4 e 7,63 versus seis vezes.
- Contribuição: atendida.
- Apresentação: boa, com pequenos problemas editoriais.
- Referências: adequadas e atualizadas no manuscrito.
- Escopo: atendido.
- Integridade verificável: indeterminada nesta rodada, pois os artefatos externos não foram inspecionados.

Condições mínimas para inverter a decisão:

1. Corrigir e recomputar todas as instâncias da decomposição de variância.
2. Tratar Friedman–Nemenyi como análise nominal/descritiva ou obter unidades verdadeiramente independentes.
3. Substituir “unambiguous labels” por descrição administrativa precisa.
4. Alinhar o Resumo à natureza semiautomática e não validada da triagem SSIM.
5. Demonstrar, em ambiente limpo, que os artefatos recompõem as análises centrais.

**Probabilidade calibrada de aceitação da versão atual:** 35%.  
**Após as correções textuais, numéricas e de verificabilidade, mantendo o escopo condicional:** aproximadamente 75–85%.  
**Confiança neste parecer:** 91%.

Cinco ações com maior efeito/custo:

1. Corrigir a decomposição e o fator de crescimento em todas as seções — **reanálise existente + texto**, <2 h.
2. Reescrever a linguagem Friedman–Nemenyi como nominal/descritiva — **texto**, <1 h.
3. Corrigir “unambiguous labels”, “flags outliers” e “automated pipeline” — **texto**, <1 h.
4. Executar uma recomposição limpa das tabelas e publicar comandos/checksums — **reanálise existente**, 1–3 h.
5. Para uma futura afirmação inferencial mais forte, adquirir campanhas independentes e cruzar partições fixas com repetições de treinamento — **novo experimento**, custo elevado; não é necessário para aceitação se o artigo permanecer explicitamente condicional e de configuração única.