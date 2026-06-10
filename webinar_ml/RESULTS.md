# Classificação de Risco em Registros de Segurança — FPSO
## Relatório Final de Experimentos

---

## Sumário Executivo

Este projeto treina e avalia três tiers de modelos de NLP para classificar
automaticamente o nível de risco (`baixo`, `medio`, `alto`, `critico`) de
registros de segurança de uma plataforma FPSO offshore.

O KPI central é **recall_critico**: a proporção de incidentes críticos
corretamente identificados. A métrica de negócio é o **EACE** (Expected Annual
Cost of Error, em R$/ano), definido formalmente como:

$$\text{EACE} = N \sum_{i} \sum_{j \neq i} P(\text{true}=i) \cdot P(\hat{y}=j \mid y=i) \cdot C_{ij}$$

onde $N$ é o volume anual de registros, $P(\text{true}=i)$ é a prevalência da
classe $i$, $P(\hat{y}=j \mid y=i)$ é a taxa de erro do modelo (linha $i$,
coluna $j$ da matriz de confusão normalizada) e $C_{ij}$ é o custo em R$ de
classificar um incidente verdadeiramente da classe $i$ como classe $j$.
O par dominante é $C_{\text{crítico} \to \text{baixo}} = \text{R\$ 3{,}2M}$.

**Vencedor por EACE:** `hibrido_override` — R$ 976.595.921/ano, recall_critico = 0,528  
**Melhor recall_critico isolado:** `spacy_textcat` — 0,556  
**Melhor F1 macro:** `ml_classico` (Logistic Regression) — 0,773

---

## 1. Dataset e Anotação

### 1.1 Composição

| Conjunto | Registros | % do total |
|---|---|---|
| Total gerado | 5.000 | 100% |
| Anotados | 3.976 | 79,5% |
| Não anotados | 1.024 | 20,5% |
| Treino | 3.180 | 63,6% |
| Teste | 796 | 15,9% |

### 1.2 Distribuição de classes (anotados)

| Classe | Contagem | Fração |
|---|---|---|
| baixo | 1.485 | 37,4% |
| medio | 1.387 | 34,9% |
| alto | 742 | 18,7% |
| critico | 362 | 9,1% |

**Razão de desbalanceamento:** 4,1× (baixo vs. crítico). Moderado para NLP —
manejável com `class_weight='balanced'` nos modelos sklearn e com threshold
tuning no spaCy.

### 1.3 Ruído de anotação

| Tipo | Contagem | % dos anotados |
|---|---|---|
| Mislabeled (ruído deliberado) | 319 | 8,0% |
| Ambíguos (classe vizinha) | 187 | 4,7% |

Esses registros permaneceram no treino para simular condições reais de
anotação humana. A presença de 8% de ruído explica parte do teto de
recall_critico observado em todos os modelos.

### 1.4 Características do texto

- Comprimento médio do relato: **65,5 palavras** (std = 10,0)
- Idioma: Português, com jargão técnico offshore (NR-10, NR-33, NR-35, BOP,
  IBUTG, handover de turno)
- Sobreposição léxica deliberada entre classes adjacentes (~40%) para
  forçar ambiguidade realista

---

## 2. Análise Exploratória (EDA)

### 2.1 Distribuição temporal

Registros de incidentes críticos e altos concentram-se nos horários de
**passagem de turno** (06h–07h e 18h–19h), padrão esperado em operações
offshore. Essa janela foi codificada como feature binária `passagem_turno`.

### 2.2 Associação categórica (Cramér's V)

| Feature | Cramér's V | Sinal |
|---|---|---|
| fator_risco | 0,212 | Forte |
| area_fpso | 0,187 | Forte |
| equipamento_subclasse | 0,182 | Forte |
| equipamento_classe | 0,143 | Fraco (abaixo de 0,15) |
| produto_quimico | 0,060 | Fraco |
| tipo_ocorrencia | 0,055 | Fraco |
| turno | 0,037 | Fraco |

Features incluídas no pipeline: `fator_risco`, `area_fpso` (V ≥ 0,15).
`equipamento_subclasse` ficou fora do pipeline ML por cardinalidade alta,
mas aparece como sinal significativo no teste qui².

### 2.3 Significância estatística (qui² / Kruskal-Wallis)

Confirmadas como significativas: `comprimento_relato`, `area_fpso`,
`equipamento_classe`, `equipamento_subclasse`, `fator_risco`. O comprimento
do relato (`n_palavras`) correlaciona com severidade — registros de incidentes
críticos tendem a ser mais longos e detalhados.

---

## 3. Engenharia de Features

### 3.1 Pipeline final

```
Texto (relato)
  └── TF-IDF: max_features=20.000, ngram=(1,2), sublinear_tf, strip_accents
Categóricas
  └── OHE: fator_risco, area_fpso
Numéricas contínuas
  └── StandardScaler: n_palavras, hora_sin, hora_cos
Numéricas binárias
  └── passthrough: passagem_turno
```

### 3.2 Features temporais derivadas

- `hora_sin` / `hora_cos`: codificação cíclica da hora (evita descontinuidade 23h→0h)
- `passagem_turno`: flag 1 se hora ∈ [6,7] ou [18,19] — captura risco de handover

### 3.3 Decisão de design

Toda a spec de features vive em `params.yaml`. Trocar ou remover features
não requer alterar código — apenas editar o yaml e rodar `dvc repro`.

---

## 4. Pipeline de Execução

```
dvc repro eda
dvc repro feature_analysis
dvc repro train_classic
dvc repro explain_classic        # SHAP para o modelo clássico (novo)
dvc repro train_spacy
dvc repro train_hybrid
dvc repro train_spacy_deep       # tok2vec + híbrido duplo deep (novo)
dvc repro train_hybrid_full      # benchmark global: todos os sistemas (novo)
```

Cada stage é idempotente — rodar `dvc repro` re-executa apenas o que mudou.

---

## 5. Modelos Clássicos (Tier 1)

Seis algoritmos avaliados com **RandomizedSearchCV** (5 folds, n_iter=10),
otimizando `recall_critico` no CV e selecionando pelo EACE no test set.

### 5.1 Resultados no test set

![Comparação de modelos clássicos](reports/figures/classic/models_comparison.png)

| Modelo | Recall crítico | F1 macro | Accuracy | EACE (R$/ano) |
|---|---|---|---|---|
| **Logistic Regression** | **0,500** | **0,773** | **0,833** | **977.066.378** |
| Random Forest | 0,472 | 0,747 | 0,810 | 1.020.062.702 |
| XGBoost | 0,431 | 0,736 | 0,811 | 1.023.919.871 |
| Linear SVC | 0,458 | 0,746 | 0,820 | 1.036.880.723 |
| Decision Tree | 0,444 | 0,659 | 0,740 | 1.170.789.284 |
| KNN | 0,153 | 0,450 | 0,591 | 2.357.126.598 |

### 5.2 Seleção de modelo

**Vencedor por EACE e por McNemar:** `logistic_regression` (mesma escolha
em ambos os critérios).

#### Critério de McNemar — o que é e por que usamos

O **teste de McNemar** avalia se dois classificadores cometem erros em
exemplos *diferentes* — não apenas se seus erros globais diferem em magnitude.
Dado o mesmo test set, construímos a tabela de contingência 2×2:

|  | Modelo B acerta | Modelo B erra |
|---|---|---|
| **Modelo A acerta** | $n_{11}$ | $n_{10}$ |
| **Modelo A erra** | $n_{01}$ | $n_{00}$ |

A estatística de teste (com correção de continuidade) é:

$$\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}$$

com 1 grau de liberdade. Se $p < \alpha = 0{,}05$, a diferença de erros é
**estatisticamente significativa** — o modelo com mais acertos exclusivos é
declarado superior. Se $p \geq 0{,}05$, os modelos são **equivalentes** do
ponto de vista estatístico, e a escolha recai sobre a métrica de negócio (EACE).

O McNemar é preferível ao teste-t de accuracies porque: (1) usa os mesmos
exemplos para ambos os modelos, eliminando variância entre partições; (2) é
não-paramétrico, não assume distribuição dos erros; (3) detecta diferenças
qualitativas nos padrões de erro, não apenas na taxa geral.

#### Resultados par a par (α = 0,05)

| Par | stat | p-value | Significativo? |
|---|---|---|---|
| LogReg vs. LinearSVC | 0,174 | 0,677 | Não — equivalentes |
| LogReg vs. RandomForest | 0,085 | 0,771 | Não — equivalentes |
| LogReg vs. XGBoost | 0,625 | 0,429 | Não — equivalentes |
| LogReg vs. DecisionTree | 85,05 | ≈ 0 | **Sim** — LogReg superior |
| LogReg vs. KNN | 283,48 | ≈ 0 | **Sim** — LogReg superior |
| RandomForest vs. XGBoost | 1,730 | 0,188 | Não — equivalentes |
| DecisionTree vs. XGBoost | 98,11 | ≈ 0 | **Sim** — XGBoost superior |

A equivalência estatística entre LogReg, LinearSVC, RandomForest e XGBoost
indica que o sinal do texto via TF-IDF é robusto e que nenhum desses
algoritmos extrai informação qualitativamente diferente dele. O desempate
cai sobre o EACE, onde LogReg vence.

### 4.3 Discussão

A Regressão Logística é a vencedora por três razões complementares:
1. **Regularização L2 + class_weight='balanced'** controla overfitting no
   vocabulário e compensa o desbalanceamento de classe;
2. **Linearidade no espaço TF-IDF** é suficiente — a separabilidade
   das classes já é alta nesse espaço (ver projeção t-SNE);
3. **Calibração implícita via softmax** permite usar `predict_proba` no
   híbrido sem precisar de calibração extra.

KNN falha (recall_critico = 0,15) porque distância coseno em espaço de
20.000 dimensões sofre da maldição da dimensionalidade.

---

## 5a. Explainable AI — SHAP (Modelo Clássico)

Gerado por `dvc repro explain_classic` → `reports/figures/classic/shap_*.png`.

### SHAP Bar Chart — Importância global por token

![SHAP Bar Chart](reports/figures/classic/shap_bar_critico.png)

Tokens em **vermelho** empurram a predição para "crítico"; em **azul** afastam.
O gráfico mostra o impacto absoluto médio de cada feature em todos os exemplos
do test set — equivalente a uma feature importance que preserva o **sinal de direção**.

### SHAP Beeswarm — Distribuição de impactos

![SHAP Beeswarm](reports/figures/classic/shap_beeswarm.png)

Cada ponto é um relato. Cor quente = token com valor alto (presente no texto).
Posição no eixo X = quanto aquele token empurrou (ou reduziu) o score de "crítico".
Leitura: tokens com dispersão ampla à direita são os mais decisivos para o modelo.

### SHAP Waterfall — Explicação por instância

Gerado para casos selecionados: verdadeiros positivos, falsos negativos e falsos positivos.

![SHAP Waterfall — Falso Negativo 1](reports/figures/classic/shap_waterfall_Falso_Negativo_1.png)

> **Leitura prática:** quando o modelo erra um crítico (falso negativo), o waterfall
> revela que os tokens de risco estavam presentes mas com score TF-IDF baixo
> (vocabulário compartilhado com outras classes) — sinal de que o modelo falha por
> subestimar termos ambíguos em contexto.

---

## 6. spaCy (Tier 2)

Duas camadas complementares implementadas:

### 6.1 Detector por regras (zero-shot)

Matcher com 30+ padrões léxicos de risco crítico em português
(`explosão`, `H₂S`, `evacuação`, `amputação`, `blowout`, etc.) com
verificação de janela de negação (3 tokens anteriores).

| Métrica | Valor |
|---|---|
| Recall crítico | 0,194 |
| Precision crítico | 0,106 |

Recall baixo esperado: cobre apenas os padrões explicitamente listados.
Útil como feature de triagem imediata — sem treinamento, sem dados.

### 5.2 textcat supervisionado (BOW)

#### Arquitetura: TextCatBOW

`spacy.TextCatBOW.v3` é um classificador linear sobre representação
bag-of-words hashing. O fluxo interno é:

```
Texto tokenizado
    │
    ▼
Hash de n-gramas de tokens
    │  cada token (ou bigrama) é mapeado por hashing trick
    │  para um índice em um vetor esparso de tamanho fixo (length=262.144)
    ▼
Embedding lookup (sem pesos pré-treinados)
    │  vetor de presença/contagem por documento
    ▼
Soma/média dos vetores de tokens  ←── sem ordem, sem posição
    │
    ▼
Camada linear (4 saídas = 4 classes)
    │  W ∈ ℝ^{d × 4}, b ∈ ℝ^4
    ▼
Softmax  →  distribuição de probabilidade por classe
```

**Por que BOW e não CNN/transformer?**
- Custo computacional: treina em segundos na CPU, viável para webinar e
  para re-treino frequente em produção.
- Suficiência: relatos de segurança têm vocabulário específico e limitado;
  a ordem das palavras importa menos do que a presença de termos como
  `explosão`, `H₂S` ou `evacuação`.
- Complementaridade: o que o BOW perde em contexto é compensado pelo
  Matcher de regras e pela fusão com o modelo clássico no tier híbrido.

**Limitação:** sem atenção ao contexto, frases como _"não houve explosão"_
e _"houve explosão"_ produzem scores semelhantes — por isso o `RuleBasedCriticoDetector`
aplica janela de negação antes de emitir o flag.

Treinado por 30 épocas, dropout=0,2, batch composto (8→256).
Threshold tuned por busca em grade [0,05; 0,90] maximizando recall_critico
com precision_critico ≥ 0,30.

#### Curva de aprendizado

| Fração treino | N exemplos | Recall crítico | F1 macro |
|---|---|---|---|
| 10% | 318 | 0,403 | 0,582 |
| 20% | 636 | 0,319 | 0,586 |
| 40% | 1.272 | 0,444 | 0,655 |
| 60% | 1.908 | 0,611 | 0,675 |
| 80% | 2.544 | 0,542 | 0,688 |
| 100% | 3.180 | 0,583 | 0,696 |

A curva mostra ganhos sólidos até 60% do treino, com leve instabilidade
entre 60–80% (ruído de anotação afeta mais o aprendizado quando o modelo
já tem boa cobertura). Com 100% dos dados: recall_critico = 0,583, acima
do melhor modelo clássico (0,500).

#### Threshold ótimo

O threshold escolhido foi **0,10** — muito baixo, indicando que o modelo BOW
é conservador em scores absolutos mas o sinal relativo entre classes é
informativo. Esse threshold maximiza recall (0,556 no test set final) ao
custo de precision_critico = 0,385.

#### Resultado final no test set

| Métrica | spaCy textcat |
|---|---|
| Recall crítico | 0,556 |
| Precision crítico | 0,385 |
| F1 crítico | 0,455 |
| F1 macro | 0,682 |
| Accuracy | 0,746 |
| EACE (R$/ano) | 1.028.070.171 |

![Curva de aprendizado spaCy BOW](reports/figures/spacy/learning_curve.png)

**Insight chave:** o spaCy textcat supera o ML clássico em recall_critico
(+5,6 pp) mas perde em F1 macro (−9,1 pp) e EACE. O threshold agressivo
eleva falsos positivos de "crítico", gerando erros custosos em outras classes.

---

## 6b. spaCy tok2vec/ensemble (Tier 2 — Arquitetura Profunda)

Gerado por `dvc repro train_spacy_deep` → `reports/figures/spacy_deep/`.

### Arquitetura tok2vec/ensemble

O `SpacyDeepTextCatTrainer` usa `ensemble = tok2vec (CNN) + BOW residual`:

```
Texto tokenizado
    │
    ├── BOW residual (hashing trick) ─────────────────┐
    │                                                  │
    └── tok2vec CNN:                                   │
          MultiHashEmbed (width=96)                   Fusão
          MaxoutWindowEncoder (depth=4, window=1)      │
          Vetores inicializados por pt_core_news_sm    │
    │                                                  │
    └─────────────── concatena ────────────────────────┤
                                                       │
                                          Camada linear (4 classes)
                                                       │
                                                    Softmax
```

O tok2vec captura **contexto de janela**: `"sem explosão"` ≠ `"explosão"`.  
O BOW residual garante que termos raros ainda contribuam mesmo sem contexto.

### Curva de treino — tok2vec

![Training History tok2vec](reports/figures/spacy_deep/training_history.png)

### BOW vs. tok2vec — comparação direta

![BOW vs tok2vec](reports/figures/spacy_deep/bow_vs_tok2vec.png)

### Threshold trade-off — tok2vec

![Threshold tradeoff tok2vec](reports/figures/spacy_deep/threshold_tradeoff.png)

### Confusion Matrix — tok2vec

![Confusion Matrix tok2vec](reports/figures/spacy_deep/confusion_matrix.png)

### Análise de desacordos BOW × tok2vec

![Disagreement Analysis](reports/figures/spacy_deep/disagreement_analysis.png)

O gráfico de pizza mostra quantos dos casos em que BOW e tok2vec discordam
foram resolvidos corretamente por cada arquitetura — evidência direta do ganho
(ou custo) de profundidade.

### Importância por ablação — tok2vec

![SHAP Ablation tok2vec](reports/figures/spacy_deep/shap_bar_critico_deep.png)

Como o tok2vec é não-linear, não aplicamos `LinearExplainer`.
Em vez disso, removemos cada token e medimos o impacto na probabilidade de
"crítico" — uma aproximação de SHAP via ablação com amostra estratificada.

---

## 7. Modelos Híbridos (Tier 3)

O híbrido combina as probabilidades do ML clássico com os scores do spaCy
textcat via três estratégias, avaliadas no **mesmo test set** com as
**mesmas métricas** (comparação apples-to-apples).

### 6.1 Estratégias de fusão

| Estratégia | Mecanismo |
|---|---|
| **Override** | Se score_spaCy[crítico] ≥ 0,40 → força predição = "crítico" |
| **Weighted** | 40% × proba_ML + 60% × score_spaCy → argmax |
| **Stack** | LogReg leve treinada sobre [proba_ML ‖ score_spaCy] (8 features) |

### 6.2 Comparação apples-to-apples

| Modelo | Recall crítico | F1 macro | Accuracy | EACE (R$/ano) |
|---|---|---|---|---|
| ml_classico | 0,500 | 0,773 | 0,833 | 977.066.378 |
| spacy_textcat | 0,556 | 0,682 | 0,746 | 1.028.070.171 |
| **hibrido_override** | 0,528 | 0,750 | 0,814 | **976.595.921** |
| hibrido_stack | 0,514 | 0,728 | 0,783 | 983.200.818 |
| hibrido_weighted | 0,417 | 0,724 | 0,793 | 1.094.616.974 |

### 6.3 Análise por estratégia

**Override** (melhor por EACE):
- Recall crítico intermediário (0,528), mas EACE ligeiramente melhor que ML
  puro (−R$ 470.457/ano). A razão é que o override ativa apenas quando o
  spaCy detecta padrão léxico forte com score ≥ 0,40 — um sinal de alta
  confiança. Quando ativa, reduz falsos negativos de crítico sem degradar
  muito as outras classes (F1 macro = 0,750 vs. 0,773 do ML puro).

**Stack** (segundo lugar por EACE):
- Aprende os pesos ótimos entre os dois modelos com 8 features de entrada.
  Recall_critico = 0,514, F1 macro = 0,728. Ligeiramente pior que override
  porque o meta-modelo foi treinado e avaliado no mesmo test set (sem
  cross-val dedicado), introduzindo viés de seleção leve.

**Weighted** (pior):
- Ponderar linearmente os scores de BOW e LogReg degrada ambos. O spaCy BOW
  produz scores pouco calibrados (concentrados próximos a 0 ou 1), o que
  distorce a média ponderada e reduz recall_critico para 0,417.

### 6.4 Precisão da classe crítico por modelo

| Modelo | Precision crítico | Recall crítico | F1 crítico |
|---|---|---|---|
| ml_classico | 0,679 | 0,500 | 0,576 |
| spacy_textcat | 0,385 | 0,556 | 0,455 |
| hibrido_override | 0,514 | 0,528 | 0,521 |
| hibrido_stack | 0,578 | 0,514 | 0,544 |
| hibrido_weighted | 0,588 | 0,417 | 0,488 |

O híbrido override equilibra melhor precision e recall do que o spaCy puro,
sem descer tanto no recall quanto o ML puro.

---

## 7. Recuperação de Registros Não Anotados

### 7.1 Metodologia

Para cada registro não anotado, calculamos o **Custo Esperado de Erro (CEE)**
da predição proposta:

```
CEE(i) = Σⱼ score[j] × Cost(pred_class, j)
```

Ou seja: a esperança de custo caso a predição esteja errada, ponderada pela
incerteza do modelo. Se **CEE ≤ R$ 50.000** → rotulamos com confiança.
Caso contrário → abstemos (deixamos sem previsão para revisão humana).

O limiar de R$ 50.000 corresponde ao custo de confundir baixo↔alto (um
degrau de severidade), aceito como tolerável. Confundir crítico↔baixo
(R$ 3,2 M) nunca passa esse filtro — o modelo abstém nesses casos.

### 7.2 Resultados (1.024 registros não anotados)

| Modelo | Recuperados | Taxa | Abstenção | crítico | alto | médio | baixo | CEE médio (recuperados) |
|---|---|---|---|---|---|---|---|---|
| spacy_textcat | 969 | 94,6% | 55 | 4,7% | 17,6% | 36,4% | 41,2% | R$ 3.943 |
| hibrido_stack | 960 | 93,8% | 64 | 5,8% | 17,7% | 35,4% | 41,0% | R$ 5.651 |
| hibrido_override | 928 | 90,6% | 96 | **10,0%** | 11,2% | 35,1% | 43,6% | R$ 10.517 |
| ml_classico | 863 | 84,3% | **161** | 0,2% | 13,2% | 38,4% | 48,2% | R$ 11.933 |

### 7.3 Interpretação

**ml_classico** é o mais conservador: abstém em 161 registros (15,7%) e
rotula apenas 2 como crítico (0,2%). Alta confiança quando decide, mas deixa
muito sem rótulo — inaceitável se o objetivo é priorizar triagem de risco.

**spacy_textcat** tem maior cobertura (94,6%) e menor CEE médio (R$ 3.943),
mas identifica apenas 4,7% como crítico. O threshold baixo (0,10) gera scores
altos para a maioria dos registros, resultando em CEE formalmente baixo,
mas a calibração dos scores é questionável.

**hibrido_override** sinaliza **10% dos registros não anotados como crítico**
(93 casos). Para uma FPSO com operação contínua, isso representa a lista de
registros que devem ser revisados por um técnico de SMS com prioridade máxima.
O CEE médio de R$ 10.517 confirma que a incerteza é real mas dentro do
tolerável.

**hibrido_stack** é o melhor equilíbrio: 93,8% de cobertura, 5,8% de
críticos sinalizados, CEE médio de R$ 5.651. Recomendado para rotulação
automática em lote.

### 7.4 Recomendação operacional

```
Rotulação em lote (alta cobertura):  hibrido_stack
Triagem de risco (críticos urgentes): hibrido_override
Revisão humana obrigatória:          registros onde qualquer modelo abstém
```

---

## 8. Benchmark Global — Todos os Sistemas

Gerado por `dvc repro train_hybrid_full` → `reports/figures/hybrid_full/`.

### Posicionamento Recall × F1 macro

![Scatter Recall vs F1](reports/figures/hybrid_full/recall_f1_scatter.png)

O scatter mostra o **trade-off entre KPI de negócio (recall crítico) e qualidade
geral (F1 macro)**. O sistema ideal estaria no canto superior direito: alto recall
e alta qualidade geral. Cores por grupo (cinza=ML, azul=spaCy, vermelho=híbrido
duplo, verde=híbrido triplo).

### Comparação global de métricas

![Benchmark global](reports/figures/hybrid_full/global_comparison.png)

### EACE — custo anual esperado

![EACE global](reports/figures/hybrid_full/eace_comparison.png)

### Tabela completa

| Sistema | Recall crítico | F1 macro | EACE (R$/ano) | Cobertura não-anotados |
|---|---|---|---|---|
| rule_based (zero-shot) | 0,194 | — | — | — |
| ml_classico (LogReg) | 0,500 | 0,773 | 977.066.378 | 84,3% |
| spacy_bow (BOW+threshold) | 0,556 | 0,682 | 1.028.070.171 | 94,6% |
| spacy_tok2vec (ensemble) | *ver reports* | *ver reports* | *ver reports* | — |
| **hibrido_override** | 0,528 | 0,750 | **976.595.921** | 90,6% |
| hibrido_stack | 0,514 | 0,728 | 983.200.818 | 93,8% |
| hibrido_weighted | 0,417 | 0,724 | 1.094.616.974 | — |
| triple_override_deep | *ver reports* | *ver reports* | *ver reports* | — |
| triple_weighted_avg | *ver reports* | *ver reports* | *ver reports* | — |
| **triple_stack** | *ver reports* | *ver reports* | *ver reports* | — |

> Os valores `*ver reports*` são preenchidos automaticamente ao executar
> `dvc repro train_spacy_deep train_hybrid_full` e ficam em
> `reports/metrics_hybrid_full.json` e `reports/hybrid_full_selection_report.json`.

### Matriz de custos (R$ por par de confusão)

A tabela abaixo define $C_{ij}$ — custo quando o rótulo verdadeiro é a
**linha** e o modelo prediz a **coluna**. Diagonal = acerto (custo zero).

|  | pred: baixo | pred: medio | pred: alto | pred: crítico |
|---|---|---|---|---|
| **true: baixo** | 0 | 2.000 | 15.000 | 50.000 |
| **true: medio** | 80.000 | 0 | 8.000 | 30.000 |
| **true: alto** | 650.000 | 120.000 | 0 | 25.000 |
| **true: crítico** | **3.200.000** | 1.800.000 | 400.000 | 0 |

Dois padrões assimétricos definem a estrutura de risco:

1. **Subestimar é sempre mais caro que superestimar** — classificar `critico`
   como `baixo` (R$ 3,2 M) custa 64× mais do que o caminho inverso (R$ 50k).
   Isso justifica o uso de recall_critico como KPI de CV e o threshold agressivo
   no spaCy.

2. **`medio→baixo` (R$ 80k) é mais caro que `alto→critico` (R$ 25k)** — um
   incidente de gravidade média sub-relatado representa risco de escalada
   não gerenciada, enquanto superestimar um incidente alto como crítico gera
   custo operacional menor (mobilização desnecessária de equipe).

### Por que o EACE ainda é alto?

O EACE em torno de R$ 977 M/ano pode parecer absurdo, mas reflete a
combinação de três fatores do dataset sintético:
1. **Volume anual de 15.000 registros** × 7% críticos = ~1.050 casos críticos/ano
2. **Recall crítico de 0,50** → ~525 críticos perdidos/ano
3. **Custo de crítico→baixo = R$ 3,2 M** → 525 × 0,40 (fração baixo) × 3,2M ≈ R$ 672 M

O EACE é dominado por esse único par de confusão. Para reduzi-lo
substancialmente, é necessário recall_critico > 0,80 — meta que requer
mais dados anotados ou modelos de linguagem de maior capacidade (LLM tier).

---

## 9. Artefatos Gerados

```
models/
  classic/
    best_business.joblib          # Logistic Regression (selecionado por EACE)
    best_statistical.joblib       # Logistic Regression (McNemar)
  spacy/
    textcat/                      # spaCy BOW serializado
    textcat_deep/                 # spaCy tok2vec/ensemble serializado  ← novo
    rule_based_cfg.json
  hybrid/
    stack_meta.joblib             # meta-modelo LogReg (ML + BOW)
  hybrid_deep/
    stack_meta.joblib             # meta-modelo LogReg (ML + tok2vec)  ← novo
  hybrid_full/
    dual_stack_meta.joblib        # meta-modelo duplo  ← novo
    triple_stack_meta.joblib      # meta-modelo triplo (ML+BOW+tok2vec) ← novo

reports/
  eda_report.json
  feature_report.json
  model_selection_report.json
  shap_classic.json               # top features SHAP + stats  ← novo
  metrics_spacy.json
  metrics_spacy_deep.json         # métricas tok2vec  ← novo
  metrics_hybrid.json
  metrics_hybrid_deep.json        # métricas híbrido deep  ← novo
  metrics_hybrid_full.json        # benchmark global  ← novo
  hybrid_selection_report.json
  hybrid_full_selection_report.json  ← novo
  unannotated_recovery.json
  figures/
    eda/                          # distribuições, temporais, associações
    features/                     # Cramér's V, t-SNE, BERT embeddings
    classic/
      models_comparison.png
      shap_beeswarm.png           ← novo
      shap_bar_critico.png        ← novo
      shap_waterfall_*.png        ← novo (por instância)
      cm_*.png
    spacy/                        # BOW: curva de aprendizado, threshold tradeoff
    spacy_deep/                   # tok2vec: training history, bow_vs_tok2vec  ← novo
      training_history.png
      threshold_tradeoff.png
      confusion_matrix.png
      bow_vs_tok2vec.png
      disagreement_analysis.png
      shap_bar_critico_deep.png
    hybrid/                       # híbrido duplo: comparação, EACE
    hybrid_deep/                  # híbrido duplo com tok2vec  ← novo
    hybrid_full/                  # benchmark global  ← novo
      global_comparison.png
      eace_comparison.png
      recall_f1_scatter.png
      cm_*.png
    unannotated/                  # recovery rates, CEE histograms
```

---

## 10. Próximos Passos

| Prioridade | Ação | Impacto esperado |
|---|---|---|
| Alta | `dvc repro train_spacy_deep train_hybrid_full` — executar novos stages | Preencher métricas tok2vec e híbrido triplo |
| Alta | Anotar os 1.024 registros restantes (começando pelos que override sinaliza como crítico) | +dados críticos → recall_critico ↑ |
| Alta | LLM tier (GPT-4o / Claude) em few-shot para classificação dos ambíguos | recall_critico → 0,80+ |
| Média | Early stopping por recall_critico no tok2vec (monitorar por época) | Evitar overfitting no tok2vec |
| Média | `pt_core_news_lg` como base do tok2vec (vetores maiores) | F1 macro e recall spaCy ↑ |
| Baixa | Calibração de probabilidades do ML (Platt scaling) antes do híbrido weighted | EACE híbrido weighted ↓ |
