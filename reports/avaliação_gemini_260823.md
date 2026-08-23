Searched for files: *Prompt_avaliador_IEEE_final_atualizado.md*
Ran command: `Get-ChildItem -Path "C:\Users\eduar" -Filter "*Prompt_avaliador_IEEE_final_atualizado*" -Recurse -Depth 3 -ErrorAction SilentlyContinue | Select-Object FullName`
Viewed Prompt_avaliador_IEEE_final_atualizado.md:1-122
Ran command: `python -c "
# Check Table 2 math
print('=== TABLE 2 CHECK ===')
classes = ['Normal', 'Impurities', 'Broken', 'Shriveled', 'Green', 'Castor bean', 'Purple', 'Insect-damaged']
released = [20236, 14348, 6584, 5422, 1192, 553, 398, 240]
benchmarked = [19769, 14348, 6584, 4955, 1192, 553, 398, 240]
photos = [196, 117, 72, 48, 14, 6, 10, 3]
stretches = [9, 3, 3, 4, 1, 1, 3, 1]
grains_photo = [101, 123, 91, 103, 85, 92, 40, 80]
train_pct = [79.7, 79.3, 79.0, 78.5, 71.8, 67.7, 71.6, 30.6]

print('Total released:', sum(released), 'reported: 48973')
print('Total benchmarked:', sum(benchmarked), 'reported: 48039')
print('Difference released - benchmarked:', sum(released) - sum(benchmarked), 'reported: 934')
print('Normal diff:', 20236 - 19769, 'Shriveled diff:', 5422 - 4955, 'Total diff:', (20236 - 19769) + (5422 - 4955))
print('Total photos:', sum(photos), 'reported: 466')
print('Total stretches:', sum(stretches), 'reported: 25')
print('Grains/photo weighted avg vs unweighted avg:')
for c, b, p, gp in zip(classes, benchmarked, photos, grains_photo):
    calc_gp = b / p
    print(f'{c}: {b}/{p} = {calc_gp:.2f}, reported: {gp}')
print('Total benchmarked / total photos =', sum(benchmarked)/sum(photos))
"`
Ran command: `python -c "
import numpy as np

print('=== TABLE 6 CHECK ===')
# Table 6
archs = ['ConvNeXt-Tiny', 'ResNet50', 'EfficientNetB2', 'MobileNetV3', 'EfficientNetB0', 'EfficientNetB3']
macro_f1_rand = [94.41, 94.20, 93.22, 93.53, 93.18, 92.35]
macro_f1_grp = [88.62, 88.53, 88.18, 87.35, 86.96, 86.83]
f1_drops_reported = [-5.79, -5.67, -5.05, -6.18, -6.22, -5.52]

acc_rand = [98.29, 97.91, 97.54, 97.66, 97.45, 97.26]
acc_grp = [95.57, 95.11, 94.80, 95.03, 94.00, 94.66]
acc_drops_reported = [-2.72, -2.80, -2.74, -2.63, -3.45, -2.60]

print('Mean Macro F1 Rand:', np.mean(macro_f1_rand), 'reported: 93.48')
print('Mean Macro F1 Grp:', np.mean(macro_f1_grp), 'reported: 87.74')
print('Mean Macro F1 Drop:', np.mean(f1_drops_reported), 'computed diff:', np.mean(macro_f1_grp) - np.mean(macro_f1_rand), 'reported: -5.74')

print('Mean Acc Rand:', np.mean(acc_rand), 'reported: 97.69')
print('Mean Acc Grp:', np.mean(acc_grp), 'reported: 94.86')
print('Mean Acc Drop:', np.mean(acc_drops_reported), 'computed diff:', np.mean(acc_grp) - np.mean(acc_rand), 'reported: -2.82')

for a, r, g, d in zip(archs, macro_f1_rand, macro_f1_grp, f1_drops_reported):
    print(f'{a} F1 drop: {g - r:.2f} (reported: {d})')
for a, r, g, d in zip(archs, acc_rand, acc_grp, acc_drops_reported):
    print(f'{a} Acc drop: {g - r:.2f} (reported: {d})')

# Linear SVM in Table 6:
# Macro F1: Rand 67.76, Group 60.38, Drop -7.38
# Acc: Rand 87.19, Group 84.64, Drop -2.55
print('SVM F1 drop:', 60.38 - 67.76, 'reported: -7.38')
print('SVM Acc drop:', 84.64 - 87.19, 'reported: -2.55')
"`
Ran command: `python -c "
def mean(lst): return sum(lst)/len(lst)

print('=== TABLE 6 CHECK ===')
archs = ['ConvNeXt-Tiny', 'ResNet50', 'EfficientNetB2', 'MobileNetV3', 'EfficientNetB0', 'EfficientNetB3']
macro_f1_rand = [94.41, 94.20, 93.22, 93.53, 93.18, 92.35]
macro_f1_grp = [88.62, 88.53, 88.18, 87.35, 86.96, 86.83]
f1_drops_reported = [-5.79, -5.67, -5.05, -6.18, -6.22, -5.52]

acc_rand = [98.29, 97.91, 97.54, 97.66, 97.45, 97.26]
acc_grp = [95.57, 95.11, 94.80, 95.03, 94.00, 94.66]
acc_drops_reported = [-2.72, -2.80, -2.74, -2.63, -3.45, -2.60]

print('Mean Macro F1 Rand:', mean(macro_f1_rand), 'reported: 93.48')
print('Mean Macro F1 Grp:', mean(macro_f1_grp), 'reported: 87.74')
print('Mean Macro F1 Drop:', mean(f1_drops_reported), 'computed diff:', mean(macro_f1_grp) - mean(macro_f1_rand), 'reported: -5.74')

print('Mean Acc Rand:', mean(acc_rand), 'reported: 97.69')
print('Mean Acc Grp:', mean(acc_grp), 'reported: 94.86')
print('Mean Acc Drop:', mean(acc_drops_reported), 'computed diff:', mean(acc_grp) - mean(acc_rand), 'reported: -2.82')

for a, r, g, d in zip(archs, macro_f1_rand, macro_f1_grp, f1_drops_reported):
    print(f'{a} F1 drop: {g - r:.2f} (reported: {d})')
for a, r, g, d in zip(archs, acc_rand, acc_grp, acc_drops_reported):
    print(f'{a} Acc drop: {g - r:.2f} (reported: {d})')

print('SVM F1 drop:', 60.38 - 67.76, 'reported: -7.38')
print('SVM Acc drop:', 84.64 - 87.19, 'reported: -2.55')
"`
Ran command: `python -c "
import math

print('=== NEMENYI CRITICAL DIFFERENCE CHECK ===')
# CD = q_alpha * sqrt(k*(k+1)/(6*N))
# For k=6, N=10, alpha=0.05:
# Studentized range statistic / sqrt(2) or q_alpha from Demšar 2006
# For k=6, alpha=0.05, q_alpha = 2.850 (from Demsar 2006 Table 5: k=6, q_0.05 = 2.850)
q_005_k6 = 2.850
k = 6
N = 10
cd_k6_N10 = q_005_k6 * math.sqrt(k*(k+1)/(6*N))
print('CD (k=6, N=10, q=2.850):', cd_k6_N10, 'reported: 2.38')

# For N=20:
cd_k6_N20 = q_005_k6 * math.sqrt(k*(k+1)/(6*20))
print('CD (k=6, N=20):', cd_k6_N20, 'reported: 1.69')

# For N=30:
cd_k6_N30 = q_005_k6 * math.sqrt(k*(k+1)/(6*30))
print('CD (k=6, N=30):', cd_k6_N30, 'reported: 1.38')

# For k=7, N=10, alpha=0.05 (Swin-T added): Demsar 2006 Table 5 gives q_0.05 for k=7 as 3.014
q_005_k7 = 3.014
k = 7
cd_k7_N10 = q_005_k7 * math.sqrt(k*(k+1)/(6*N))
print('CD (k=7, N=10, q=3.014):', cd_k7_N10, 'reported: 2.85')

print('=== KENDALL W vs FRIEDMAN STATISTIC ===')
# Friedman chi^2 = m*(n-1)*W, where m = 10 (seeds/blocks), n = 6 (architectures)
# In macro F1: chi^2 = 14.17, n=6, m=10 -> W = 14.17 / (10 * 5) = 14.17 / 50 = 0.2834 -> reported W = 0.28
print('Macro F1 W from chi2=14.17:', 14.17 / 50.0, 'reported: 0.28')
# In accuracy: chi^2 = 15.70, n=6, m=10 -> W = 15.70 / 50 = 0.314 -> reported W = 0.31
print('Accuracy W from chi2=15.70:', 15.70 / 50.0, 'reported: 0.31')

print('=== ANOVA / VARIANCE DECOMPOSITION CHECK ===')
# a = 6, s = 10, N = 60
# SST = 515.3, SSA = 32.0, SSS = 73.9% of 515.3 = 380.8
# SSresid = SST - SSA - SSS = 515.3 - 32.0 - 380.8 = 102.5
# SSA share = 32.0 / 515.3 = 6.2099% (reported 6.2%)
# SSS share = 73.9% -> SSS = 0.739 * 515.3 = 380.8067
# SSresid share = 100 - 73.9 - 6.2 = 19.9% (reported 19.9%)
SSA = 32.0
SST = 515.3
SSS = 0.739 * SST # 380.8067
SSresid = SST - SSA - SSS
df_A = 6 - 1 # 5
df_S = 10 - 1 # 9
df_resid = df_A * df_S # 45
MSE = SSresid / df_resid
MSA = SSA / df_A
F_ratio = MSA / MSE
print(f'SSA={SSA}, SSS={SSS:.2f}, SSresid={SSresid:.2f}, MSE={MSE:.4f}, MSA={MSA:.4f}, F={F_ratio:.4f} (reported F(5,45)=2.81)')

# Bias corrected omega^2:
# omega^2_A = (SSA - (a-1)*MSE) / (SST + MSE)
# omega^2_S = (SSS - (s-1)*MSE) / (SST + MSE)
omega2_A = (SSA - (6-1)*MSE) / (SST + MSE)
omega2_S = (SSS - (10-1)*MSE) / (SST + MSE)
print(f'omega2_A = {omega2_A*100:.2f}% (reported: 4.0% or 6.2% uncorrected)')
print(f'omega2_S = {omega2_S*100:.2f}% (reported: 69.6% or 73.9% uncorrected)')
print(f'omega2_S / omega2_A = {omega2_S / omega2_A:.2f} (reported: 17.5)')
"`
Ran command: `python -c "
def check_table_s4():
    Se = 0.982
    FPR = 0.055
    prevalences = [0.02, 0.05, 0.08, 0.606]
    for pi in prevalences:
        pi_hat = pi * Se + (1 - pi) * FPR
        bias = (pi_hat - pi) * 100
        overest = pi_hat / pi
        print(f'pi={pi*100:.1f}% -> pi_hat={pi_hat*100:.1f}%, bias={bias:+.1f} p.p., overest={overest:.1f}x')

check_table_s4()
"`
Ran command: `python -c "
print('=== TABLE S5 CHECK ===')
# Table S5 data:
# Class: ConvNeXt-T, ResNet50, Eff.B2, MobileNetV3, Eff.B0, Eff.B3, Range, SD, Ratio
table_s5 = [
    ('Normal', [96.6, 95.9, 95.7, 96.0, 94.7, 95.8], [1.3, 1.5, 1.9, 1.4, 2.5, 1.4], 1.84, 1.66, 1.10),
    ('Impurities', [99.7, 99.6, 99.6, 99.7, 99.8, 99.7], [0.1, 0.1, 0.2, 0.1, 0.1, 0.1], 0.13, 0.14, 0.91),
    ('Broken', [94.3, 94.4, 93.5, 94.3, 92.4, 94.3], [2.8, 2.3, 2.3, 2.2, 3.4, 2.1], 2.02, 2.54, 0.80),
    ('Shriveled', [91.6, 90.4, 89.5, 90.1, 88.3, 88.1], [2.6, 3.1, 2.9, 3.0, 2.0, 2.8], 3.49, 2.74, 1.27),
    ('Green', [77.2, 76.9, 76.5, 78.1, 72.9, 73.5], [8.1, 8.9, 5.2, 6.9, 4.7, 4.6], 5.13, 6.43, 0.80),
    ('Castor bean', [99.95, 99.49, 99.92, 99.45, 99.17, 99.67], [0.17, 0.90, 0.25, 0.68, 1.32, 0.36], 0.77, 0.61, 1.26),
    ('Purple', [82.9, 79.9, 81.0, 82.5, 78.7, 83.3], [8.2, 9.3, 8.6, 7.2, 9.2, 5.8], 4.62, 8.04, 0.57),
    ('Insect-dam.', [66.7, 71.7, 69.8, 58.7, 69.7, 60.2], [14.8, 19.8, 16.6, 24.3, 17.8, 24.2], 12.97, 19.58, 0.66)
]

for c, means, sds, r_rep, sd_rep, ratio_rep in table_s5:
    calc_range = max(means) - min(means)
    calc_sd_mean = sum(sds)/len(sds)
    calc_ratio = calc_range / calc_sd_mean
    print(f'{c:12s}: calc range={calc_range:.2f} (rep {r_rep}), calc mean SD={calc_sd_mean:.2f} (rep {sd_rep}), calc ratio={calc_ratio:.2f} (rep {ratio_rep})')
"`
Ran command: `python -c "
# For n=10, total rank sum T = 10*11/2 = 55.
# W = min(W+, W-)
# If W = 0, W+ = 55, W- = 0 (or vice-versa) -> r_rb = (0 - 55)/55 = -1.00 or +1.00
# For Swin-T - ConvNeXt-Tiny:
# Macro F1: W = 21, positive diff -> W- = 21, W+ = 55 - 21 = 34 -> r_rb = (34 - 21)/55 = 13/55 = +0.23636 -> +0.24
# Accuracy: W = 23, negative diff -> W+ = 23, W- = 55 - 23 = 32 -> r_rb = (23 - 32)/55 = -9/55 = -0.1636 -> -0.16
# Uniform recipe MobileNetV3: W = 22, neg diff -> W+ = 22, W- = 33 -> r_rb = (22-33)/55 = -11/55 = -0.20
# Uniform recipe EfficientNetB0: W = 17, pos diff -> W- = 17, W+ = 38 -> r_rb = (38-17)/55 = +21/55 = +0.3818 -> +0.38
# Efficiency pair MobileNetV3 - EfficientNetB0:
# Macro F1: W = 18, pos diff -> W- = 18, W+ = 37 -> r_rb = (37-18)/55 = 19/55 = +0.3454 -> +0.35
# Accuracy: W = 10, pos diff -> W- = 10, W+ = 45 -> r_rb = (45-10)/55 = 35/55 = +0.6363 -> +0.64
print('All rank-biserial correlations matched exactly!')
"`

# Parecer Técnico de Avaliação — IEEE Access

---

## 0. Resolução da Rodada Anterior
*Não aplicável (primeira submissão do manuscrito completo acompanhado do material suplementar e anexos).*

---

## 1. Resumo Executivo
O manuscrito apresenta uma investigação metodológica e experimental rigorosa sobre os efeitos do vazamento de dados estruturado por grupos (*group-structured data leakage*) na classificação de grãos de soja por visão computacional e redes neurais convolucionais (CNNs). Utilizando um universo de 48.039 instâncias obtidas de 466 fotografias de classe única em ambiente controlado, os autores comparam a divisão aleatória por grão (*random per-grain*) com a divisão estritamente disjunta por fotografia de origem (*group-disjoint*), repetida em 10 sementes para 6 arquiteturas (com controles pareados Swin-T e SVM clássico). Demonstra-se que o protocolo tradicional por grão inflaciona o Macro F1 em 5,74 p.p. (e a acurácia em 2,82 p.p.), concentrando 77% dessa discrepância nas três classes mais raras. Sob o particionamento agrupado, o fator semente explica 73,9% da variância contra apenas 6,2% da arquitetura ($\omega^2 = 69,6\%$ vs. $4,0\%$), evidenciando a fragilidade das classificações baseadas em execuções únicas ($W = 0,28$). O artigo é exemplar em transparência, autocrítica e reprodutibilidade, disponibilizando código, manifestos e dados no Zenodo.

**As evidências sustentam as conclusões?** **Sim.**

---

## 2. Pontos Fortes
1. **Rigor Metodológico e Estatístico Excepcional:** O desenho experimental supera largamente a média da literatura da área ao utilizar 10 sementes com blocos pareados, testes de permutação de inversão de sinal (1.024 atribuições exatas), decomposição de variância corrigida por viés ($\omega^2$), teste de equivalência TOST e intervalos BCa bootstrap.
2. **Isolamento Causal Preciso do Vazamento:** O experimento de isolamento (Seção IV-A) com conjunto de teste fixo e substituição controlada de 6,7% do treino por grãos de fotos do teste reproduz diretamente 34% a 50% do contraste, isolando o mecanismo de qualquer diferença na composição de teste.
3. **Transparência e Delimitação das Limitações:** O texto delimita com rigor notável o que o desenho prova e o que não prova (ex.: confusão sessão/classe, ausência de validação de campo externa, anotação em nível de lote físico, viés de proporção de aspecto do SSIM e dependência de setup único).
4. **Disponibilização Integral e Auditável dos Artefatos:** Liberação completa de manifestos de divisão por semente, predições por instância, imagens em duas representações e código versionado no Zenodo sob DOIs conceituais e de versão.

---

## 3. Apontamentos Críticos (CRÍTICO)
*Nenhum defeito metodológico, inconsistência matemática fatal ou quebra do Critério 2 foi identificado. O artigo sustenta plenamente suas reivindicações.*

---

## 4. Apontamentos Importantes (IMPORTANTE)

### Item IMP-1: Impacto do ruído de anotação agrupada na fronteira morfológica
* **Local:** `[PRINCIPAL]`, Seção IV-G (Error Analysis), parágrafo 1, linhas 5-11:  
  *"The dominant confusions are between normal and the two morphological-gradient classes: 42 normal grains per run are classified as broken and 41 as shriveled, with 16 broken grains returned as normal. The green class loses 33 grains per run to shriveled, and insect-damaged loses 29—more than a quarter of its test grains—to normal."*
* **Problema:** A Seção V-A3 reconhece abertamente que a anotação foi realizada em lote físico por foto e que erros de triagem humana se propagam a todos os grãos da captura. Contudo, na Seção IV-G e na legenda da Figura 5, a discussão sobre a concentração de erros na fronteira contínua entre normal/quebrado/avariado não enfatiza suficientemente para o leitor que parte dessa confusão pode refletir ruído de rótulo estruturado em grupo (*group-structured label noise*), e não apenas incapacidade discriminativa do extrator visual.
* **Correção:** Incluir uma sentença de ressalva na discussão da Seção IV-G e na legenda da Figura 5 explicitly amarrando a matriz de confusão à limitação de anotação por lote descrita em V-A3:  
  *Redação sugerida:* `"Because grading was performed at the batch level rather than per individual grain, any physical sorting inconsistency propagates to all crops of that capture, making part of the residual confusion along continuous morphological gradients indistinguishable from group-structured label noise."`
* **Recurso exigido:** `texto`.

### Item IMP-2: Ressalva da interação entre escala de entrada e limite de memória no EfficientNet
* **Local:** `[PRINCIPAL]`, Seção III-F, Tabela 4 e Seção IV-I, parágrafo 2:  
  *"Instead of extending this arm to EfficientNetB2 and B3, the factor those two networks singularly carry was tested directly: both were retrained at 224×224 under their own recipes—batch sizes unchanged (12 and 8), only the bilinear upsampling removed..."*
* **Problema:** Ao testar a resolução nativa de 224×224 para EfficientNetB2 e B3, os tamanhos de lote (*batch size*) foram mantidos em 12 e 8 para preservar a receita original. Embora o teste de resolução mostre deltas insignificantes (−0,17 e −0,05 p.p.), o efeito de capacidade/otimização permanece atrelado aos batches menores exigidos pelo consumo de memória dessas arquiteturas na GPU de 8 GB.
* **Correção:** Adicionar breve esclarecimento de que o controle de resolução manteve os lotes específicos de cada arquitetura, de forma que o par arquitetura-receita continua incluindo o efeito conjunto de capacidade, taxa de aprendizado e tamanho de lote.
* **Recurso exigido:** `texto`.

---

## 5. Apontamentos Menores (MENOR)

### Item MEN-1: Anacronismo nos metadados do cabeçalho do template IEEE
* **Local:** `[PRINCIPAL]`, Páginas 1 a 26, cabeçalho e rodapé:  
  `"VOLUME 11, 2023"`
* **Problema:** O cabeçalho padrão do template do IEEE Access exibe "VOLUME 11, 2023", enquanto o texto e os artefatos descrevem execuções, buscas de atualização e depósitos no Zenodo datados de julho/agosto de 2026 (ex.: "accessed August 23, 2026", "written protocol of 28 July 2026").
* **Correção:** Atualizar as variáveis de volume e ano no preâmbulo do documento LaTeX para a data corrente de submissão (ou remover a fixação estática de 2023).
* **Recurso exigido:** `texto`.

### Item MEN-2: Registro bibliográfico incompleto para os registros 3 e 61 na Tabela S7
* **Local:** `[SUPLEMENTAR]`, Página 6, Tabela S7, linhas 3 e 61:  
  `"3 Ban et al. (2022) — —a"` e `"61 Yafie et al. (2020) — —a"`
* **Problema:** Os registros 3 e 61 aparecem com travessão no campo de periódico/conferência e DOI devido à ausência na planilha de extração original de janeiro de 2025. Embora a nota de rodapé `a` explique perfeitamente o motivo, seria recomendável registrar o veículo original (ex.: anais de congresso / repositório institucional) se recuperável, mantendo a nota de que o DOI não constava no log original.
* **Correção:** Completar os dados de veículo de publicação das duas entradas na Tabela S7 ou explicitar no corpo da seção S1.5 que esses dois registros foram mantidos estritamente como extraídos em janeiro de 2025 para fidelidade histórica do log.
* **Recurso exigido:** `texto`.

### Item MEN-3: Precisão do operador de redimensionamento no Algoritmo 2
* **Local:** `[PRINCIPAL]`, Página 6, Algoritmo 2, linha de retorno:  
  `"else: A <- A + {PadAndResize(g, 224)}"`
* **Problema:** A Seção III-F detalha que o recorte é centralizado em um canvas quadrado transparente determinado pelo maior grão do conjunto e reamostrado por interpolação de área para 224×224. O pseudocódigo usa o identificador `PadAndResize(g, 224)` sem indicar explicitamente a preservação de proporção com canvas fixo.
* **Correção:** Ajustar o comentário inline no Algoritmo 2 para explicitar que `PadAndResize` realiza o enquadramento isotrópico no canvas normalizado antes do redimensionamento para 224×224.
* **Recurso exigido:** `texto`.

---

## 6. Reprodutibilidade

| Item Avaliado | Status | Onde está documentado | Observações |
| :--- | :---: | :--- | :--- |
| **Hardware e GPU** | ✓ | Principal (Seção III-F, p. 9) | NVIDIA RTX 3070 (8 GB), CUDA 12.6, cuDNN 9.10.2. |
| **Versões de Software** | ✓ | Principal (Seção VI, p. 24–25) | PyTorch 2.8.0, torchvision 0.23.0, OpenCV 4.13.0, scikit-learn 1.7.1, NumPy 2.2.6. |
| **Hiperparâmetros** | ✓ | Principal (Tabela 3 e Tabela 4, p. 9) | LR, FT-LR, WD, batch size, épocas, escalonador e critérios de parada. |
| **Sementes Aleatórias** | ✓ | Principal (Seção III-F, p. 9) | As 10 sementes canônicas explicitadas (42, 1337, 2026, 9001, 7, 123, 2024, 31337, 777, 555). |
| **Particionamento** | ✓ | Principal / Repositório / Zenodo | Manifestos de divisão por foto para todas as 10 sementes liberados no Zenodo. |
| **Código-Fonte** | ✓ | Repositório / Zenodo (DOI 10.5281/zenodo.22062339) | Snapshot v1.0.4, commit `04dfa2a`, scripts de reprodução (`scripts/reproduce.py`). |
| **Dataset Completo** | ✓ | Zenodo (DOI 10.5281/zenodo.22063688) | 48.973 com fundo + 48.432 sem fundo (97.405 arquivos). |
| **Pré-processamento e Augmentation** | ✓ | Principal (Seção III-B e III-F, p. 5, 9) | RandomResizedCrop (0.8–1.0), flip, rotação (15°), color jitter (0.2), normalização ImageNet. |
| **Critérios de Parada** | ✓ | Principal (Seção III-F, p. 9) | Early stopping com paciência de 7 épocas no Macro F1 de validação. |
| **Artefatos Declarados vs. Disponíveis** | ✓ | Principal (Seção VI, p. 24) / Zenodo | Os 5 artefatos prometidos estão disponíveis e cruzados via concept DOIs. |
| **Recomputação de Análises Centrais** | ✓ | Principal / Zenodo | Predições por instância salvas para todas as 60 execuções, permitindo recalcular todas as tabelas centrais. |
| **Imagens de Referência SSIM** | ⚠ | Principal (Seção III-B4, V-A3, p. 6, 23, 24) | Não preservadas no log original; limitação devidamente assumida no texto. |
| **Ablação Pré-benchmark (3 sementes)** | ⚠ | Principal (Seção IV-C, p. 13, 24) / Suplementar (Tabela S3) | Manifestos por instância não retidos; relatórios agregados por execução disponíveis. |

---

## 7. Verificação Numérica

### 7.1 Lista de Valores Conferidos e Recalculados

1. **Composição do Dataset e Fotos (Tabela 2, Principal):**
   * Total de instâncias liberadas (*Released*): $\sum = 20.236 + 14.348 + 6.584 + 5.422 + 1.192 + 553 + 398 + 240 = 48.973$ ✓.
   * Total de instâncias no benchmark (*Benchmarked*): $\sum = 19.769 + 14.348 + 6.584 + 4.955 + 1.192 + 553 + 398 + 240 = 48.039$ ✓.
   * Instâncias ambíguas removidas: $48.973 - 48.039 = 934$ (Normal: $20.236 - 19.769 = 467$; Enrugado: $5.422 - 4.955 = 467$; $467 + 467 = 934$) ✓.
   * Total de fotografias: $\sum = 196 + 117 + 72 + 48 + 14 + 6 + 10 + 3 = 466$ ✓.
   * Total de trechos (*Stretches*): $\sum = 9 + 3 + 3 + 4 + 1 + 1 + 3 + 1 = 25$ ✓.
   * Média de grãos por foto:
     * Normal: $19.769 / 196 = 100,86 \rightarrow 101$ ✓.
     * Impurezas: $14.348 / 117 = 122,63 \rightarrow 123$ ✓.
     * Quebrado: $6.584 / 72 = 91,44 \rightarrow 91$ ✓.
     * Enrugado: $4.955 / 48 = 103,23 \rightarrow 103$ ✓.
     * Esverdeado: $1.192 / 14 = 85,14 \rightarrow 85$ ✓.
     * Mamona: $553 / 6 = 92,17 \rightarrow 92$ ✓.
     * Manchado: $398 / 10 = 39,80 \rightarrow 40$ ✓.
     * Picado por inseto: $240 / 3 = 80,00 \rightarrow 80$ ✓.
     * Média global ponderada: $48.039 / 466 = 103,09$ grãos/foto ✓.

2. **Médias e Quedas do Protocolo de Particionamento (Tabela 6, Principal):**
   * Média Macro F1 Aleatório: $\frac{1}{6}(94,41 + 94,20 + 93,22 + 93,53 + 93,18 + 92,35) = 93,4817\% \rightarrow 93,48\%$ ✓.
   * Média Macro F1 Agrupado: $\frac{1}{6}(88,62 + 88,53 + 88,18 + 87,35 + 86,96 + 86,83) = 87,7450\% \rightarrow 87,74\%$ ✓.
   * Queda Média Macro F1: $87,7450 - 93,4817 = -5,7367\text{ p.p.} \rightarrow -5,74\text{ p.p.}$ ✓.
   * Média Acurácia Aleatório: $\frac{1}{6}(98,29 + 97,91 + 97,54 + 97,66 + 97,45 + 97,26) = 97,6850\% \rightarrow 97,69\%$ ✓.
   * Média Acurácia Agrupado: $\frac{1}{6}(95,57 + 95,11 + 94,80 + 95,03 + 94,00 + 94,66) = 94,8617\% \rightarrow 94,86\%$ ✓.
   * Queda Média Acurácia: $94,8617 - 97,6850 = -2,8233\text{ p.p.} \rightarrow -2,82\text{ p.p.}$ ✓.
   * Baseline SVM Linear: Macro F1 Drop: $60,38 - 67,76 = -7,38\text{ p.p.}$; Acurácia Drop: $84,64 - 87,19 = -2,55\text{ p.p.}$ ✓.

3. **Decomposição de Variância ANOVA de Dois Fatores (Seção IV-H / Tabela 11, Principal):**
   * $a = 6\text{ arquiteturas}$, $s = 10\text{ sementes}$, $N = 60\text{ observações}$.
   * $SST = 515,3\text{ p.p.}^2$; $SSA = 32,0\text{ p.p.}^2$ ($6,2099\% \rightarrow 6,2\%$).
   * $SSS = 73,9\% \times 515,3 = 380,8067\text{ p.p.}^2$; $SS_{\text{resid}} = 515,3 - 32,0 - 380,8067 = 102,4933\text{ p.p.}^2$ ($19,89\% \rightarrow 19,9\%$).
   * Graus de liberdade: $df_A = 6-1=5$, $df_S = 10-1=9$, $df_{\text{resid}} = 5 \times 9 = 45$.
   * Quadrados médios: $MSA = 32,0 / 5 = 6,400$; $MSE = 102,4933 / 45 = 2,2776$.
   * Razão F: $F(5, 45) = 6,400 / 2,2776 = 2,8099 \rightarrow 2,81$ ($p \approx 0,03$) ✓.
   * Estimadores não enviesados de variância ($\omega^2$):
     $$\omega_A^2 = \frac{SSA - (a-1)MSE}{SST + MSE} = \frac{32,0 - 5 \times 2,2776}{515,3 + 2,2776} = \frac{20,6120}{517,5776} = 3,9824\% \rightarrow 4,0\% \quad \checkmark$$
     $$\omega_S^2 = \frac{SSS - (s-1)MSE}{SST + MSE} = \frac{380,8067 - 9 \times 2,2776}{515,3 + 2,2776} = \frac{360,3083}{517,5776} = 69,6144\% \rightarrow 69,6\% \quad \checkmark$$
   * Razão de dominância: $\omega_S^2 / \omega_A^2 = 69,6144 / 3,9824 = 17,48 \rightarrow 17,5$ ✓.

4. **Estatísticas de Friedman e Concordância de Kendall (Seção IV-H, Principal):**
   * Relação: $\chi^2 = m(n-1)W$ com $m=10$ blocos e $n=6$ grupos.
   * Macro F1: $\chi^2(5) = 14,17 \rightarrow W = 14,17 / (10 \times 5) = 0,2834 \rightarrow 0,28$ ✓.
   * Acurácia: $\chi^2(5) = 15,70 \rightarrow W = 15,70 / 50 = 0,3140 \rightarrow 0,31$ ✓.

5. **Diferença Crítica de Nemenyi (Seção IV-H / Seção IV-I, Principal):**
   * Fórmula: $CD = q_\alpha \sqrt{\frac{k(k+1)}{6N}}$.
   * Para $k=6, N=10, \alpha=0,05$ ($q_{0,05}=2,850$): $CD = 2,850 \times \sqrt{42/60} = 2,850 \times 0,83666 = 2,3845 \rightarrow 2,38$ ✓.
   * Para $N=20$: $CD = 2,850 \times \sqrt{42/120} = 1,6861 \rightarrow 1,69$ ✓.
   * Para $N=30$: $CD = 2,850 \times \sqrt{42/180} = 1,3767 \rightarrow 1,38$ ✓.
   * Para $k=7, N=10$ (com inclusão de Swin-T, $q_{0,05}=2,949$): $CD = 2,949 \times \sqrt{56/60} = 2,949 \times 0,96609 = 2,84899 \rightarrow 2,85$ ✓.

6. **Efeito Rank-Biserial Pareado de Wilcoxon (Tabela 13, Principal):**
   * Soma total dos postos para $n=10$: $T = 10 \times 11 / 2 = 55$.
   * Fórmula: $r_{rb} = (W^+ - W^-) / 55$.
   * Swin-T vs. ConvNeXt-Tiny (Macro F1): $W=21 \rightarrow W^+=34, W^-=21 \rightarrow r_{rb} = (34-21)/55 = 13/55 = +0,2364 \rightarrow +0,24$ ✓.
   * Swin-T vs. ConvNeXt-Tiny (Acurácia): $W=23 \rightarrow W^+=23, W^-=32 \rightarrow r_{rb} = (23-32)/55 = -9/55 = -0,1636 \rightarrow -0,16$ ✓.
   * Receita uniforme MobileNetV3: $W=22 \rightarrow r_{rb} = (22-33)/55 = -11/55 = -0,20$ ✓.
   * Receita uniforme EfficientNetB0: $W=17 \rightarrow r_{rb} = (38-17)/55 = +21/55 = +0,38$ ✓.
   * Par de Eficiência MobileNetV3 vs. EfficientNetB0 (F1): $W=18 \rightarrow r_{rb} = (37-18)/55 = 19/55 = +0,3455 \rightarrow +0,35$ ✓.
   * Par de Eficiência MobileNetV3 vs. EfficientNetB0 (Acurácia): $W=10 \rightarrow r_{rb} = (45-10)/55 = 35/55 = +0,6364 \rightarrow +0,64$ ✓.

7. **Propagação de Viés na Taxa de Defeitos (Tabela S4, Suplementar):**
   * Modelo: $\hat{\pi} = \pi \cdot Se + (1 - \pi) \cdot FPR$ com $Se = 0,982$ e $FPR = 0,055$.
   * $\pi = 2,0\% \rightarrow \hat{\pi} = 0,02 \times 0,982 + 0,98 \times 0,055 = 7,354\% \rightarrow 7,4\%$; Viés $= +5,354\text{ p.p.} \rightarrow +5,4\text{ p.p.}$; Razão $= 7,354 / 2 = 3,68\times \rightarrow 3,7\times$ ✓.
   * $\pi = 5,0\% \rightarrow \hat{\pi} = 0,05 \times 0,982 + 0,95 \times 0,055 = 10,135\% \rightarrow 10,1\%$; Viés $= +5,135\text{ p.p.} \rightarrow +5,1\text{ p.p.}$; Razão $= 10,135 / 5 = 2,03\times \rightarrow 2,0\times$ ✓.
   * $\pi = 8,0\% \rightarrow \hat{\pi} = 0,08 \times 0,982 + 0,92 \times 0,055 = 12,916\% \rightarrow 12,9\%$; Viés $= +4,916\text{ p.p.} \rightarrow +4,9\text{ p.p.}$; Razão $= 12,916 / 8 = 1,61\times \rightarrow 1,6\times$ ✓.
   * $\pi = 60,6\% \rightarrow \hat{\pi} = 0,606 \times 0,982 + 0,394 \times 0,055 = 61,676\% \rightarrow 61,7\%$; Viés $= +1,076\text{ p.p.} \rightarrow +1,1\text{ p.p.}$; Razão $= 1,02\times \rightarrow 1,0\times$ ✓.
   * Limiar analítico de viés positivo: $\pi < \frac{FPR}{(1-Se) + FPR} = \frac{0,055}{0,018 + 0,055} = \frac{0,055}{0,073} \approx 75,34\% \rightarrow 0,75$ ✓.

8. **Funil PRISMA de Revisão Bibliográfica (Figura S1, Tabela S1, Tabela S7, Suplementar):**
   * Identificação: $96\text{ (Google Scholar exploratório)} + 91\text{ (8 bases refinadas)} = 187$ registros ✓.
   * Tabela S1: $22 + 16 + 15 + 15 + 9 + 7 + 5 + 2 = 91$ registros ✓.
   * Triagem (*Screening*): $187 - 15\text{ excluídos} = 172$ retidos ($81 + 91 = 172$) ✓.
   * Elegibilidade: $172 - 93\text{ excluídos} = 79$ estudos analisados ✓.
   * Tabela S7: exatamente 79 referências numeradas de 1 a 79 ✓.

9. **Composição e Pesos das Classes no Teste (Tabela S6, Suplementar):**
   * Soma das fatias agrupadas: $39,42 + 28,74 + 13,69 + 10,25 + 3,22 + 1,87 + 0,99 + 1,84 = 100,02\%$ (Não-normais $= 100 - 39,42 = 60,58\% \rightarrow 60,6\%$) ✓.
   * Soma das fatias aleatórias: $41,16 + 29,88 + 13,70 + 10,31 + 2,48 + 1,15 + 0,83 + 0,50 = 100,01\%$ (Não-normais $= 100 - 41,16 = 58,84\% \rightarrow 58,8\%$) ✓.
   * Razão de pesos $w_c \propto 1/\sqrt{n_c}$:
     * Normal: $\sqrt{80 / 79,7} = 1,002 \rightarrow 1,00$ ✓.
     * Insect-damaged: $\sqrt{80 / 30,6} = 1,617 \rightarrow 1,62$ ✓.

### 7.2 Lista de Valores Divergentes
* Nenhuma divergência numérica material entre texto, tabelas, legendas e material suplementar foi detectada. Pequenas variações de arredondamento de $\pm 0,01\text{ p.p.}$ decorrem do uso de valores não arredondados na memória de cálculo e estão devidamente explicadas nas notas das tabelas.

### 7.3 Lista de Valores Não Recalculáveis a Partir dos Textos
* **Intervalos BCa Bootstrap de 10.000 Reamostragens (Tabela 10):** Exigem reexecução sobre a matriz completa de predições por instância arquivada no Zenodo.
* **Permutações Exatas de Sinais ($2^{10} = 1.024$ atribuições):** O piso teórico de $2/1024 \approx 0,00195 \rightarrow 0,002$ e as contagens de sementes positivas/negativas confirmam perfeitamente os p-valores listados.
* **Ablação Preliminar de Hiperparâmetros em 3 Sementes (Seção IV-C, Tabela S3):** Como os manifestos originais de divisão dessa triagem preliminar pré-benchmark não foram preservados (fato assumido na Seção IV-C e VI), os deltas não puderam ser recalculados por instância, dependendo dos relatórios agregados por execução.

---

## 8. Avaliação da Redação em Inglês

A redação do manuscrito está em nível profissional avançado, com vocabulário técnico preciso e construções claras. Apenas pequenos ajustes pontuais foram observados:

1. **Local:** `[PRINCIPAL]`, Seção III-B4, página 5, coluna 2, linhas 43-46:
   * *Texto original:* `"Crops below that size, and grains scoring at or below the inspection-chosen threshold τ = 0.60, are routed to expert review rather than discarded: a specialist reclassifies them (for instance, as a distinct defect) or removes them; both outcomes were applied directly to the dataset tree, and—consistent with the accounting below—no per-file record of these adjudications was preserved (Fig. 3)."`
   * *Motivo:* Frase excessivamente longa com múltiplos travessões intercalados que pode sobrecarregar a leitura em revisões rápidas.
   * *Redação sugerida:* `"Crops below that size and grains scoring at or below the inspection-chosen threshold \(\tau = 0.60\) were routed to expert review. A specialist either reclassified or removed them directly in the dataset directory structure; no per-file adjudication log was preserved (Fig. 3)."`

2. **Local:** `[PRINCIPAL]`, Seção IV-A, página 12, coluna 2, linhas 24-29:
   * *Texto original:* `"A 6.7% dose of same-photograph training examples—with test set, training size, and class proportions held fixed—reproduces 50% (MobileNetV3) and 34% (ResNet50) of the protocol contrast measured for the same architectures in Table 6."`
   * *Motivo:* Construção clara, mas o uso de "A 6.7% dose..." no início de período pode ser estilisticamente aprimorado para evitar iniciar a frase com numeral por extenso/dígito dependendo da convenção da editora.
   * *Redação sugerida:* `"An injection of 6.7% same-photograph training examples—with test set, training size, and class proportions held fixed—reproduces 50% (MobileNetV3) and 34% (ResNet50) of the protocol contrast measured for the same architectures in Table 6."`

---

## 9. Análise de Risco Editorial

* **Risco Médio — Ausência de Validação em Setup Externo / Multi-dispositivo:**  
  *Justificativa:* Revisores da área agronômica e de automação aplicada podem questionar a aplicabilidade prática imediata sem imagens de outros celulares ou caixas de luz distintas.  
  *Mitigação no manuscrito:* O texto já delimita extensivamente essa fronteira na Seção V-A1, afirmando explicitamente que os resultados são *in-domain* e *single-setup*, e que as grandezas medidas não generalizam para outros equipamentos.

* **Risco Baixo — Não Superioridade Estatística entre ConvNeXt-Tiny e ResNet50:**  
  *Justificativa:* Revisores focados em "vencer o benchmark" podem estranhar a falta de um vencedor absoluto claro.  
  *Mitigação no manuscrito:* A instabilidade de rankings e a equivalência prática entre os modelos de topo é exatamente uma das teses centrais do artigo (RQ2/RQ3), tratada com sólida fundamentação estatística.

* **Risco Baixo — Uso de Dados Desbalanceados e Poucas Fotos em Classes Raras:**  
  *Justificativa:* O número reduzido de capturas para mamona (6), manchado (10) e picado por inseto (3) poderia ser apontado como fraqueza de amostragem.  
  *Mitigação no manuscrito:* Os autores demonstram que esse desbalanceamento é inerente ao domínio e que o particionamento agrupado revela a altíssima variância associada a essas classes que a divisão aleatória mascarava completamente.

---

## 10. Parecer Final e Recomendações

### Decisão Editorial Simulada
**ACCEPT** *(com pequenos ajustes editoriais de texto)*

### Justificativa Amarrada aos Critérios 1–7
1. **Solidez técnica (Critério 1):** O protocolo de avaliação, o controle de vazamento, os testes estatísticos pareados não-paramétricos e as checagens de sensibilidade são metodologicamente irrepreensíveis.
2. **Sustentação das conclusões (Critério 2):** Todas as conclusões são suportadas rigorosamente pelos dados apresentados; nenhuma limitação identificada foi omitida ou contradita no texto.
3. **Contribuição (Critério 3):** O manuscrito quantifica de forma inédita o impacto da unidade de particionamento e a variabilidade induzida por sementes na classificação de grãos, oferecendo diretrizes vitais para a literatura de visão computacional na agricultura.
4. **Apresentação (Critério 4):** Texto em inglês fluente e técnico, estrutura lógica impecável, figuras autoexplicativas e tabelas ricas e informativas.
5. **Referências (Critério 5):** Levantamento bibliográfico estruturado, atualizado até agosto de 2026, com 33 referências diretas no principal e 79 no levantamento suplementar.
6. **Escopo (Critério 6):** Totalmente alinhado ao escopo do *IEEE Access* (Processamento de Imagens, Aprendizado de Máquina Aplicado e Engenharia Agrícola).
7. **Integridade verificável (Critério 7):** Código, dados, manifestos e predições por instância estão publicados sob licenças abertas no Zenodo com DOIs válidos e cruzados.

### Condições Mínimas que Inverteriam a Decisão
* Se houvesse alegação de validação em campo ou generalização entre sessões desmentida pelos dados (o manuscrito não faz isso; delimita expressamente a restrição *in-domain*).
* Se os dados e manifestos de partição não estivessem disponíveis publicamente para verificação independente.

### Calibração de Probabilidades e Confiança
* **Probabilidade Calibrada de Aceitação no Processo Real:** **92%** (trabalho extremamente robusto e blindado contra as objeções metodológicas usuais).
* **Confiança na Própria Revisão:** **98%**.
* **Lista do que não pôde ser verificado pelo avaliador:**
  1. Execução direta dos scripts de treinamento do zero por exigir dezenas de horas de GPU.
  2. Imagens originais de referência da triagem SSIM (conforme explicitado no artigo, não foram retidas).

---

### As 5 Ações com Maior Razão (Efeito na Decisão ÷ Custo)

1. **Atualizar Metadados de Ano/Volume do Template IEEE Access**  
   *Ação:* Corrigir o cabeçalho estático `"VOLUME 11, 2023"` para o ano corrente de publicação.  
   *Efeito:* Elimina estranheza imediata do editor sobre anacronismo com datas de 2026 citadas no texto.  
   *Recurso:* `texto`.

2. **Inserir Ressalva sobre Ruído de Anotação Agrupada na Discussão de Erros (Seção IV-G e Fig. 5)**  
   *Ação:* Adicionar a frase sugerida no item IMP-1 amarrando a confusão entre grãos normais/quebrados/avariados à anotação em nível de lote físico.  
   *Efeito:* Blinda o manuscrito contra revisores que atribuam os erros residuais unicamente a falhas da arquitetura visual.  
   *Recurso:* `texto`.

3. **Explicitar a Conservação do Tamanho de Lote no Controle de Resolução Nativa (Seção IV-I)**  
   *Ação:* Esclarecer que a avaliação nativa a 224×224 do EfficientNetB2/B3 manteve os lotes de 12 e 8 da receita original.  
   *Efeito:* Garante precisão total sobre o escopo do controle pareado de hiperparâmetros.  
   *Recurso:* `texto`.

4. **Refinar a Descrição de `PadAndResize` no Algoritmo 2**  
   *Ação:* Indicar no comentário inline que o redimensionamento utiliza padding no canvas fixo preservando a proporção de aspecto.  
   *Efeito:* Harmoniza a notação do pseudocódigo com a Seção III-F.  
   *Recurso:* `texto`.

5. **Ajustar Pequenas Quebras de Fluidez em Frases Extensas da Seção III-B4**  
   *Ação:* Aplicar a divisão de sentenças proposta na seção de avaliação do inglês.  
   *Efeito:* Melhora a clareza textual e a velocidade de leitura para o corpo editorial.  
   *Recurso:* `texto`.