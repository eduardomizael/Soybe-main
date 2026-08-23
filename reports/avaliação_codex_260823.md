## 1. Resumo executivo

Foram avaliados o manuscrito principal de 26 páginas :codex-file-citation{path="C:\Users\eduar\Downloads\Manuscript_Partitioning_Protocol_Soybean_IEEE_Access.pdf" purpose="source"} e o material suplementar de 7 páginas :codex-file-citation{path="C:\Users\eduar\Downloads\Supplementary_Material_Systematic_Review.pdf" purpose="source"}.

Não foi fornecido parecer anterior; portanto, esta não é uma reavaliação. Também não foram fornecidos os arquivos depositados no Zenodo/GitHub, os manifests, as predições por instância nem o protocolo de 28 de julho de 2026.

O resultado central — a diferença entre particionamento por grão e por fotografia dentro deste conjunto — é tecnicamente sustentado pelos números impressos. As limitações de causalidade, sessão, campanha única, curadoria, rotulagem e validade externa são reconhecidas com força compatível no resumo, métodos, discussão e conclusão.

A auditoria não encontrou divergência aritmética entre manuscrito e suplementar. Há, entretanto, uma fragilidade estatística importante: intervalos e valores de \(p\) recebem rótulos de “95%” apesar da dependência entre as dez partições. A ressalva textual reduz a gravidade, mas não confere cobertura probabilística ao procedimento.

Não há defeito crítico demonstrado no material fornecido.

**As evidências sustentam as conclusões? Parcialmente** — sim para as conclusões descritivas condicionadas ao dataset; indeterminado para abertura, recomputabilidade integral e integridade dos artefatos externos.

## 2. Pontos fortes

- Comparação pareada no mesmo universo de 48.039 instâncias, evitando que os 934 registros ambíguos contaminem apenas um dos protocolos.
- Separação explícita entre diferença de estimandos e identificação causal do mecanismo.
- Resultados por dez seeds, com reconhecimento de que seed combina partição e aleatoriedade de treinamento.
- Experimentos complementares bem direcionados: injeção de exemplos da mesma fotografia, cruzamento partição × seed de treinamento, receita uniforme, resolução nativa e baseline clássico.
- Excelente tratamento das limitações: sessão não separada, campanha única, curadoria SSIM não validada, rotulagem em lote e ausência de avaliação externa.
- Suplementar efetivamente integrado: todas as Tabelas S1–S7 e Figuras S1–S2 têm função identificável no principal.
- Afirmações sobre inexistência de prevalência na literatura foram adequadamente enfraquecidas; o levantamento é apresentado como estruturado, não como revisão sistemática PRISMA.

## 3. Críticos

Nenhum apontamento crítico sobreviveu à releitura literal.

## 4. Importantes

### I1 — Os intervalos “95%” não têm cobertura calibrada sob dependência entre seeds

**Local:** `[PRINCIPAL]`, Seção III-H: “*They are not ten independent samples: each draws a different partition of the same 466 photographs, so a photograph that trains one seed tests another, and every pair of seeds shares most of its material.*” Mais adiante: “*every interval is a run-to-run variation interval, not a population confidence interval*”.

**Problema:** O manuscrito reconhece corretamente a dependência, mas continua apresentando intervalos t/BCa como “95%” e valores nominais de \(p\). Sem uma lei de reamostragem válida para os blocos dependentes, “95%” não possui interpretação de cobertura. Renomear o intervalo como “run-to-run variation interval” não corrige isso. Como as conclusões centrais também se apoiam em magnitude, sinais e distribuição dos pontos, o problema não as derruba, mas seria levantado por um revisor estatístico.

**Correção:** Recalcular, a partir dos dez resultados por configuração, mediana, IQR, mínimo, máximo e contagens de sinais. Remover “95%”, “confidence”, \(p\) e cruzamentos em \(\alpha=0.05\) das conclusões substantivas. Substituir a explicação geral por:

> “Because the ten seed-partition configurations reuse the same 466 photographs, inferential coverage is not calibrated. We therefore report the median, interquartile range, minimum, maximum, and sign counts across the ten realized configurations. Randomization statistics and rank thresholds, where retained, are descriptive indices only and carry no nominal population-level error rate.”

Atualizar coerentemente as legendas das Tabelas 6, 10 e 13 e da Figura 7. Para inferência populacional real, seriam necessárias campanhas independentes; isso não é necessário para manter a atual conclusão estritamente descritiva.

**Recurso exigido:** `reanálise dos dados existentes`.

## 5. Menores

### M1 — Referência cruzada incorreta na proveniência experimental

**Local:** `[PRINCIPAL]`, Seção V-A.2: “*reported as the incidental replication of Section IV-G*”.

**Problema:** A replicação incidental e o limite de ruído de execução são discutidos na Seção IV-H, não na IV-G. IV-G é a análise de erros.

**Correção:** Substituir por:

> “reported as the incidental replication discussed in Section IV-H”

**Recurso exigido:** `texto`.

### M2 — Metadados residuais do template

**Local:** `[PRINCIPAL]`, primeira página e rodapé de todas as páginas: “*Digital Object Identifier*” e “*VOLUME 11, 2023*”.

**Problema:** São placeholders incompatíveis com uma submissão nova que contém literatura e experimentos de 2026. Podem transmitir a aparência de versão publicada ou metadados incorretos.

**Correção:** Remover o DOI placeholder e o volume/ano fixos na versão submetida, deixando esses campos para a produção editorial da IEEE.

**Recurso exigido:** `texto`.

### M3 — Janela temporal da revisão não é reproduzível por data exata

**Local:** `[SUPLEMENTAR]`, Seção S1.2: “*The recency criterion (S1.4) restricts inclusion to the five years preceding it (i.e., works from 2020 onward).*”

**Problema:** “Cinco anos anteriores” e “a partir de 2020” não são equivalentes sem as datas exatas das buscas e dos limites de publicação.

**Correção:** Substituir por uma declaração com limites fechados:

> “The database searches were conducted on [insert the exact search dates]. Eligible publication dates ranged from [exact start date] through [exact final search date].”

Preencher os colchetes com o registro real das buscas.

**Recurso exigido:** `texto`.

## 6. Reprodutibilidade

| Item | Estado | Onde está documentado |
|---|---:|---|
| Hardware | ✓ | Principal, III-F e Data and Code Availability |
| Versões de software | ✓ | Principal, Data and Code Availability |
| Hiperparâmetros | ✓ | Principal, Tabelas 3–4 e III-F |
| Seeds | ✓ | Principal, III-F |
| Particionamento por fotografia | ✓ | Principal, III-C; manifests declarados |
| Pré-processamento | ✓ | Principal, III-B e III-F |
| Critério de parada/checkpoint | ✓ | Principal, III-F |
| Augmentations | ✓ | Principal, III-F |
| Baseline clássico | ✓ | Principal, III-D |
| Dataset disponibilizado | ⚠ | Declarado no principal; depósito não fornecido para inspeção |
| Código disponibilizado | ⚠ | Declarado no principal; snapshot/repositório não fornecidos |
| Manifests e predições por instância | ⚠ | Declarados; não fornecidos |
| Recomposição das análises centrais | ⚠ | `scripts/reproduce.py` é citado, mas não pôde ser executado |
| Referências SSIM originais | ✗ | Principal declara que não foram preservadas |
| Concordância entre avaliadores | ✗ | Principal declara que não foi medida e não é recuperável |
| Determinismo bit a bit | ⚠ | Não imposto; flags são declaradas como disponíveis |

## 7. Verificação numérica

### Conferido

- Tabela 2: \(48.973\) instâncias liberadas; \(48.039\) benchmarkadas; diferença \(934\); total de \(466\) fotografias.
- Variantes: \(48.973 + 48.432 = 97.405\) arquivos.
- Grãos por fotografia: \(48.039/466=103{,}09\), consistente com 103 na Tabela 2; \(48.973/470=104{,}20\), consistente com “about 104”.
- Probabilidade de uma fotografia de 104 grãos permanecer em um subconjunto: \(0{,}8^{104}+2(0{,}1^{104})=8{,}34\times10^{-11}\).
- Para 14 grãos: \(0{,}8^{14}+2(0{,}1^{14})=4{,}40\%\).
- Tabela 6, macro F1: médias impressas \(93{,}4817\%\) e \(87{,}7450\%\); diferença \(-5{,}7367\) p.p., arredondada para \(-5{,}74\).
- Tabela 6, acurácia: \(97{,}6850\%-94{,}8617\%=2{,}8233\) p.p.
- Lacunas por classe: soma \(45{,}92\); média \(45{,}92/8=5{,}74\) p.p.
- Green + purple + insect-damaged: \(77{,}22\%\) da soma; insect-damaged sozinho: \(36{,}48\%\).
- Quatro classes bem amostradas: \((2{,}54+0{,}11+2{,}43+5{,}26)/8=1{,}2925\) p.p.
- Dose do experimento de isolamento: \(2.548/37.804=6{,}74\%\).
- Frações reproduzidas: \(3{,}11/6{,}18=50{,}3\%\) e \(1{,}93/5{,}67=34{,}0\%\).
- Tabela 8: retenções \(83{,}53/88{,}53=94{,}35\%\) e \(84{,}34/87{,}35=96{,}55\%\); queda média \(-4{,}005\) p.p.
- Tabela 11: \(32{,}0/515{,}3=6{,}21\%\); \(37{,}6/305{,}2=12{,}32\%\); \(18{,}3/133{,}9=13{,}67\%\); \(28{,}3/67{,}5=41{,}93\%\).
- Crescimento de \(SST\): \(515{,}3/67{,}5=7{,}63\), corretamente descrito como “nearly eightfold”.
- Razões corrigidas: \(69{,}6/4{,}0=17{,}4\) e \(16{,}1/37{,}7=0{,}427\), compatíveis com 17,5 e 0,43 quando calculadas dos valores não arredondados.
- Nemenyi: \(2{,}85\sqrt{42/60}=2{,}384\), compatível com CD = 2,38.
- Suplementar S1: \(22+16+15+15+9+7+5+2=91\); \(187-15=172\); \(172-93=79\).
- Atualização OpenAlex: \(171/2.301=7{,}43\%\).
- Tabela S4: as quatro estimativas de prevalência fecham pela fórmula publicada.
- As participações impressas da Tabela S6 somam 100,02% e 100,01%, diferença explicada por arredondamento.
- Os valores repetidos entre principal e suplementar — Tabelas S2–S6, custos, macro F1, tamanho de checkpoints, ablação e composição — são compatíveis.

### Divergente

Nenhuma divergência numérica objetiva foi encontrada nos valores impressos.

### Não recalculável com os arquivos fornecidos

- Contagens 57/60 e 60/60 e sinais por seed.
- Intervalos t/BCa, Wilcoxon e valores das permutações sign-flip.
- Decomposição completa de variância, \(\omega^2\) não arredondados e covariâncias entre classes.
- Experimento partição × seed de treinamento e seus 40 resultados.
- Ajuste de composição baseado nas predições por instância.
- Matrizes de confusão não suprimidas, sensibilidade e FPR por seed.
- Correspondência física dos crops entre as duas variantes.
- Integridade e suficiência dos arquivos depositados.
- Busca bibliográfica, triagem dos resumos e verificação individual das obras citadas.

### Coerência das afirmações centrais

1. **Diferença de protocolo de 5,74/2,82 p.p.:** consistente no resumo, contribuições, IV-A, Tabela 6, discussão e conclusão.
2. **Ranking instável:** consistente no resumo, RQ2/RQ3, IV-B, IV-H, discussão e conclusão.
3. **73,9% seed versus 6,2% arquitetura:** consistente entre contribuição, IV-H, Tabela 11, discussão e conclusão.
4. **Fotografia necessária; sessão não resolvida:** consistente entre resumo, III-C, limitações e conclusão.
5. **Pipeline aberto/reexecutável, mas não validado:** a força é consistente entre resumo, III-B, limitações, conclusão e suplementar S3. A disponibilidade externa, entretanto, não foi verificada.

## 8. Inglês

Não identifiquei frase em inglês cuja redação, por si só, comprometa a compreensão de uma afirmação científica. O texto é denso, mas tecnicamente claro. As substituições propostas em I1 e M1 destinam-se à precisão estatística e à integridade da referência cruzada, não a correção gramatical geral.

## 9. Risco editorial

- **Alto:** DOI, dataset, código, manifests e resultados por execução não estarem acessíveis ou não recomporem as tabelas anunciadas. Isso afetaria diretamente os critérios 2 e 7.
- **Médio:** um revisor estatístico rejeitar a apresentação de intervalos e valores de \(p\) nominais sob dependência entre partições.
- **Médio:** sessão confundida com classe e ausência de campanha externa. O risco é reduzido porque o título diz “single-setup” e a limitação aparece com clareza.
- **Baixo:** protocolo bibliográfico não sistemático, sem de-duplicação e dual screening. O artigo não estima prevalência e não fundamenta a contribuição em exaustividade.
- **Baixo:** placeholders do template e referência IV-G incorreta.

## 10. Parecer final

**Decisão simulada: ACCEPT.**

A decisão é sustentada por:

1. **Solidez técnica:** o contraste entre protocolos é válido como comparação descritiva entre dois estimandos no mesmo dataset.
2. **Sustentação:** as conclusões centrais permanecem dentro do que os experimentos identificam; o manuscrito evita transformar o contraste em decomposição causal.
3. **Contribuição:** quantificação do efeito do particionamento, manifests reutilizáveis e demonstração da instabilidade dos rankings interessam ao público de visão computacional aplicada.
4. **Apresentação:** texto, tabelas e figuras são legíveis e tecnicamente organizados.
5. **Referências:** apropriadas e atualizadas até 2026, embora o levantamento não seja exaustivo.
6. **Escopo:** plenamente compatível com IEEE Access.
7. **Integridade:** internamente consistente, mas a disponibilidade externa permanece indeterminada nesta avaliação.

A decisão inverteria para **REJECT — resubmission encouraged** se:

- os depósitos citados estiverem inacessíveis, incompletos ou incapazes de recompor as análises centrais;
- as predições por instância não confirmarem 57/60, as médias da Tabela 6 ou a decomposição da Tabela 11;
- os intervalos nominais continuarem sendo usados como evidência inferencial após objeção estatística formal.

**Probabilidade estimada de aceitação no processo real:** 72%.

**Confiança nesta revisão:** 91%.

Não puderam ser verificados: Zenodo v1.0.6, snapshot de código v1.0.4, commit `04dfa2a`, repositório GitHub, protocolo de 28 de julho de 2026, manifests, predições por instância, resultados por execução, referências SSIM originais e textos completos usados na revisão bibliográfica.

As cinco ações com maior razão efeito/custo são:

1. Confirmar publicamente todos os depósitos e executar `scripts/reproduce.py` em ambiente limpo. **Recurso:** `reanálise dos dados existentes`.
2. Substituir os intervalos nominais dependentes por mediana/IQR/min–max/contagem de sinais. **Recurso:** `reanálise dos dados existentes`.
3. Corrigir IV-G → IV-H e remover DOI/volume/ano do template. **Recurso:** `texto`.
4. Registrar as datas exatas e os limites temporais da busca bibliográfica. **Recurso:** `texto`.
5. Como extensão de maior custo, realizar nova campanha com classes distribuídas entre sessões e equipamentos. **Recurso:** `novo experimento`.

O histórico anterior foi usado apenas para orientar o procedimento de auditoria; a decisão e todos os números acima foram recalculados sobre os PDFs atuais, que constituem uma versão diferente.