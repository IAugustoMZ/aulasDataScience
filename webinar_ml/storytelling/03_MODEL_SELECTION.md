# Seleção de Modelo Clássico — FPSO Safety Records
## Documentação Técnica e Analítica

> **Fonte de dados:** `reports/model_selection_report.json`  
> **Contexto:** Comparação de seis algoritmos de classificação supervisionada (Tier 1) por dois critérios complementares — métrica de negócio (EACE) e teste estatístico de McNemar — sobre o mesmo test set estratificado de 796 registros.

---

## 1. Critérios de Seleção

O processo de seleção usa dois critérios independentes que, quando concordam, validam mutuamente a escolha.

### 1.1 EACE — Expected Annual Cost of Error

$$\text{EACE} = N \sum_{i} \sum_{j \neq i} P(\text{true}=i) \cdot P(\hat{y}=j \mid y=i) \cdot C_{ij}$$

onde:
- $N$ = volume anual de registros (15.000 no dataset sintético)
- $P(\text{true}=i)$ = prevalência da classe $i$ no test set
- $P(\hat{y}=j \mid y=i)$ = taxa de erro na linha $i$, coluna $j$ da matriz de confusão normalizada
- $C_{ij}$ = custo em R\$ de classificar um incidente verdadeiramente da classe $i$ como classe $j$

A matriz de custos assimétrica é o coração do EACE:

|  | pred: baixo | pred: medio | pred: alto | pred: crítico |
|--|-------------|-------------|------------|---------------|
| **true: baixo**   | 0 | 2.000 | 15.000 | 50.000 |
| **true: medio**   | 80.000 | 0 | 8.000 | 30.000 |
| **true: alto**    | 650.000 | 120.000 | 0 | 25.000 |
| **true: crítico** | **3.200.000** | 1.800.000 | 400.000 | 0 |

O par dominante é $C_{\text{crítico} \to \text{baixo}} = \text{R\$}\,3{,}2\text{M}$, que custa **64×** mais do que o inverso ($C_{\text{baixo} \to \text{crítico}} = \text{R\$}\,50\text{k}$). Isso significa que o EACE penaliza fortemente o recall baixo na classe crítico.

### 1.2 Teste de McNemar

O teste de McNemar avalia se dois classificadores erram em **exemplos diferentes** — não apenas se suas taxas de erro globais diferem. Dado o mesmo test set, constrói-se a tabela de contingência 2×2:

|  | Modelo B acerta | Modelo B erra |
|--|-----------------|---------------|
| **Modelo A acerta** | $n_{11}$ | $n_{10}$ |
| **Modelo A erra**   | $n_{01}$ | $n_{00}$ |

A estatística de teste (com correção de continuidade de Edwards) é:

$$\chi^2 = \frac{\left(|n_{10} - n_{01}| - 1\right)^2}{n_{10} + n_{01}}$$

distribuída como $\chi^2$ com 1 grau de liberdade. Se $p < \alpha = 0{,}05$, a diferença é estatisticamente significativa e o modelo com maior $n_{10}$ (mais acertos exclusivos) é declarado superior. Se $p \geq 0{,}05$, os modelos são **equivalentes** e o desempate usa o EACE.

**Por que McNemar em vez de teste-t de acurácias?**
1. Usa os mesmos exemplos para ambos os modelos — elimina variância de particionamento.
2. É não-paramétrico — não assume distribuição dos erros.
3. Detecta diferenças qualitativas nos padrões de erro, não apenas na taxa geral.

**Resultado do processo:** `best_by_business_metric` = `logistic_regression`, `best_by_statistical_test` = `logistic_regression`, `same_winner` = `true`.

---

## 2. Ranking por EACE

| Pos. | Modelo | EACE (R\$/ano) | Recall crítico |
|------|--------|----------------|----------------|
| 1 | **logistic_regression** | **977.066.378** | **0,500** |
| 2 | random_forest | 1.020.062.702 | 0,472 |
| 3 | xgboost | 1.023.919.871 | 0,431 |
| 4 | linear_svc | 1.036.880.723 | 0,458 |
| 5 | decision_tree | 1.170.789.284 | 0,444 |
| 6 | knn | 2.357.126.598 | 0,153 |

**Análise crítica:**

A diferença entre o 1º (LogReg, R\$ 977M) e o 4º (LinearSVC, R\$ 1.037M) colocados é de apenas R\$ 60M — uma separação de ~6% sobre o baseline. Isso indica que o sinal do texto via TF-IDF é robusto e que a função de decisão linear é suficiente: qualquer modelo que consiga uma boa separação linear no espaço TF-IDF converge para resultados similares.

O KNN (R\$ 2.357M) é um outlier negativo. O recall_critico de 0,153 confirma o colapso por maldição da dimensionalidade:

$$d_{\text{coseno}}(\mathbf{x}_i, \mathbf{x}_j) \xrightarrow{d \to \infty} \text{constante}$$

Em 20.000 dimensões, todos os pontos ficam equidistantes e a vizinhança local perde significado. O modelo essencialmente chuta a classe majoritária para a maioria dos exemplos.

A Decision Tree (R\$ 1.171M) é o segundo pior, apesar de recall_critico razoável (0,444). O problema está na F1 macro baixa (0,659 vs. 0,773 do LogReg): a árvore superajusta a alguns críticos mas degrada nas outras classes — partições recursivas binárias não generalizam bem em espaço TF-IDF esparso de alta dimensão.

**Ranking por CV vs. ranking por test set:** LogReg e LinearSVC trocam de posição entre CV e test set (CV: LogReg > RF > LinearSVC; test: LogReg > RF > XGBoost > LinearSVC). Essa discrepância é esperada — o CV otimiza recall_critico internamente e o test set avalia EACE. Modelos que maximizam recall em CV podem aceitar falsos positivos que encarecem o EACE.

---

## 3. Testes McNemar Par a Par

### 3.1 Tabela completa ($\alpha = 0{,}05$)

| Par | $\chi^2$ | $p$-valor | Significativo? | Vencedor |
|-----|----------|-----------|----------------|---------|
| LogReg vs. LinearSVC | 0,174 | 0,677 | Não | LogReg (EACE) |
| LogReg vs. RandomForest | 0,085 | 0,771 | Não | LogReg (EACE) |
| LogReg vs. XGBoost | 0,625 | 0,429 | Não | LogReg (EACE) |
| **LogReg vs. DecisionTree** | **85,05** | **≈ 0** | **Sim** | **LogReg** |
| **LogReg vs. KNN** | **283,48** | **≈ 0** | **Sim** | **LogReg** |
| LinearSVC vs. RandomForest | 0,568 | 0,451 | Não | RF (EACE) |
| **LinearSVC vs. DecisionTree** | **89,80** | **≈ 0** | **Sim** | **LinearSVC** |
| LinearSVC vs. XGBoost | 0,121 | 0,728 | Não | LinearSVC (EACE) |
| **LinearSVC vs. KNN** | **283,61** | **≈ 0** | **Sim** | **LinearSVC** |
| **RandomForest vs. DecisionTree** | **86,35** | **≈ 0** | **Sim** | **RF** |
| RandomForest vs. XGBoost | 1,730 | 0,188 | Não | RF (EACE) |
| **RandomForest vs. KNN** | **273,72** | **≈ 0** | **Sim** | **RF** |
| **DecisionTree vs. XGBoost** | **98,11** | **≈ 0** | **Sim** | **XGBoost** |
| **DecisionTree vs. KNN** | **126,99** | **≈ 0** | **Sim** | **DecisionTree** |
| **XGBoost vs. KNN** | **295,10** | **≈ 0** | **Sim** | **XGBoost** |

### 3.2 Análise crítica

Os resultados revelam **dois grupos estatisticamente distintos**:

**Grupo A — equivalentes entre si (não significativos):** LogReg, LinearSVC, RandomForest, XGBoost  
**Grupo B — inferiores ao Grupo A (todos os pares significativos):** DecisionTree, KNN

Dentro do Grupo A, nenhum par tem $p < 0{,}05$, o que implica que as diferenças de erro que existem entre eles ocorrem nos **mesmos exemplos**. Esses quatro modelos erram e acertam os mesmos registros — evidência de que o teto de desempenho é imposto pela representação (TF-IDF) e pelo ruído de anotação, não pela função de decisão.

A equivalência estatística é, paradoxalmente, um resultado positivo: confirma que TF-IDF + modelo linear é uma representação estável. A robustez da representação é condição necessária para o pipeline híbrido — se o ML clássico fosse sensível à escolha de algoritmo, o meta-modelo de stacking ficaria instável.

O único par do Grupo A onde $\chi^2$ se aproxima de 1 (LogReg vs. RF = 0,085) indica quase perfeita sobreposição de erros: esses dois modelos erram exatamente os mesmos exemplos, apenas diferindo na confiança/calibração das probabilidades.

---

## 4. Justificativa da Escolha Final

**Vencedor: `logistic_regression`** — confirmado por ambos os critérios.

As três razões complementares:

1. **Regularização L2 + `class_weight='balanced'`:** controla overfitting no vocabulário esparso e compensa o desbalanceamento de classe sem exigir oversampling. O gradiente de $\ell_2$ encolhe coeficientes de tokens raros em direção a zero, evitando que hápax legômena do treino dominem a predição.

2. **Linearidade no espaço TF-IDF é suficiente:** a projeção t-SNE mostra separabilidade moderada-a-boa nesse espaço. Fronteiras de decisão não-lineares (RF, XGBoost) não adicionam ganho quantificável — confirmado pelo McNemar — e introduzem custo de interpretabilidade.

3. **Calibração implícita via softmax:** a regressão logística produz probabilidades via:

$$P(\hat{y} = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j} e^{\mathbf{w}_j^\top \mathbf{x}}}$$

Essas probabilidades são melhor calibradas que os scores de probabilidade de RF/XGBoost (que requerem Platt scaling ou isotonic regression), o que é crítico para o meta-modelo de stacking e para o cálculo do CEE nos registros não anotados.
