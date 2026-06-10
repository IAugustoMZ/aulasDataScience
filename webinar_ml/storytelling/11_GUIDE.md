# Guia de Apresentação — Webinar ML para Classificação de Risco FPSO
## Roteiro de 40 minutos | Com divisão de tempo e pontos obrigatórios

---

> **Filosofia do roteiro:** cada bloco termina com uma frase-âncora que conecta o que acabou de ser mostrado com o que vem a seguir. Não improvise a transição — use a frase exata ou uma variação próxima.

---

## Bloco 1 — O Problema de Negócio `[0:00 → 5:00]` ⏱ 5 min

**Objetivo:** fazer o público sentir o custo do problema *antes* de ver qualquer modelo.

### O que falar

**Slide de abertura (1 min):**
> "Uma plataforma FPSO gera 15 mil registros de segurança por ano. Hoje, classificar esses registros é trabalho manual: 3 mil horas de analista, 20% de backlog sem classificação, e 8% de discordância entre anotadores. Mas o problema real não é a ineficiência — é o custo do erro."

**A matriz de custos (2 min):**
- Mostre a tabela de custos do `00_BUSINESS_CASE.md` (seção 2.1).
- Destaque a célula `crítico → baixo`: **R$ 3,2 milhões por evento**.
- Diga a frase: *"Esse número inclui multa da ANP, multa ambiental, parada de produção e indenizações. É o custo de não identificar um incidente crítico."*
- Mostre o contraste: sobreclassificar um `baixo` como `crítico` custa R$ 50 mil — **64 vezes menos**.

**O número que você leva para o CFO (2 min):**
- Diga a equação simples: *"+1 ponto percentual de recall@crítico = 10,5 eventos críticos a menos por ano = R$ 33 milhões em custo esperado evitado."*
- Mostre a tabela de EACE por nível de modelo (seção 3.2 do business case).
- Termine com: *"F1 Macro é a nota do modelo. EACE é a conta do negócio. Hoje vamos construir os dois."*

**Frase de transição:**
> "Para construir qualquer modelo, precisamos primeiro entender os dados. Vamos ver o que temos."

---

## Bloco 2 — EDA e Qualidade dos Dados `[5:00 → 12:00]` ⏱ 7 min

**Objetivo:** mostrar que o dataset é realista, tem problemas reais, e que os problemas foram decisões conscientes.

### O que falar

**Visão geral rápida (1 min):**
- 5.000 registros, 4 classes, 20% não anotados, 8% com ruído deliberado.
- Diga: *"Esses números não são defeitos do dataset — são a realidade de qualquer sistema de segurança industrial."*

**Gráfico obrigatório 1 — Distribuição de classes (1,5 min):**
- Abra `reports/figures/eda/class_distribution.png`.
- Destaque: 362 críticos = apenas 9,1% do total.
- *"Com 362 exemplos críticos no treino, qualquer modelo que otimize acurácia global vai ignorar essa classe. Por isso nosso KPI principal é recall@crítico, não acurácia."*

**Gráfico obrigatório 2 — Breakdown de anotação (1,5 min):**
- Abra `reports/figures/eda/annotation_breakdown.png`.
- *"Os 8% mislabeled foram mantidos intencionalmente. Isso cria um teto real de desempenho. Um modelo perfeito ainda erraria parte dos críticos — porque o rótulo no dataset está errado."*

**Gráfico obrigatório 3 — Top tokens por classe (1 min):**
- Abra `reports/figures/eda/top_tokens_by_class.png`.
- *"Há sobreposição léxica de 40% entre classes adjacentes. 'Vazamento' aparece em incidentes médios e críticos. O modelo precisa de contexto, não só de vocabulário."*

**Gráfico obrigatório 4 — Heatmap de associação / Cramér's V (1 min):**
- Abra `reports/figures/eda/association_heatmap.png`.
- *"fator_risco tem V=0,21 — a variável estruturada com maior poder preditivo. Mas olhem: nenhuma variável estruturada passa de 0,22. O sinal está principalmente no texto."*

**Resumo rápido (1 min):**
- Dataset viável, desbalanceamento tratável, ruído real, features estruturadas auxiliares.
- Diga: *"Os 1.024 registros não anotados não são lixo. São uma exposição ao risco de R$ 88 milhões/ano — e vamos recuperá-los no final."*

**Frase de transição:**
> "Com esses dados em mãos, vamos ver quais features carregam sinal real — e como transformá-las em input para o modelo."

---

## Bloco 3 — Engenharia de Features e ML Clássico `[12:00 → 20:00]` ⏱ 8 min

**Objetivo:** mostrar a construção do pipeline Tier 1 — de features até o modelo vencedor, com justificativa de cada escolha.

### O que falar

**O pipeline de features (2 min):**
- Mostre o diagrama de features do `02_FEATURE_ANALYSIS.md` (seção 6).
- Destaque o TF-IDF com bigramas: *"'sem explosão' e 'pressão elevada' são bigramas distintos. Sem bigramas, o modelo trata os dois como iguais."*
- Destaque `passagem_turno`: *"Incidentes críticos se concentram em 06h–07h e 18h–19h — passagem de turno. Isso é conhecimento de domínio offshore codificado como uma feature binária. Simples e eficaz."*
- Abra `reports/figures/features/temporal_features.png` para ilustrar.

**Projeção t-SNE (1 min):**
- Abra `reports/figures/features/tfidf_projection.png`.
- *"Classes extremas estão separadas. Classes vizinhas se sobrepõem. É exatamente por isso que recall@crítico fica em 0,50 — não em 0,90. O teto é imposto pelos dados, não pelo algoritmo."*

**Seleção de modelo (2,5 min):**
- Mostre o ranking de EACE do `03_MODEL_SELECTION.md` (seção 2).
- Destaque que LogReg, LinearSVC, RF e XGBoost são **estatisticamente equivalentes** pelo teste de McNemar.
- *"Quatro algoritmos diferentes erram exatamente os mesmos exemplos. Isso prova que o teto é a representação — TF-IDF — não a função de decisão. Então escolhemos Regressão Logística pelo critério de explicabilidade: coeficientes globais são auditáveis."*
- Destaque o KNN com EACE de R$ 2,3 bilhões: *"Maldição da dimensionalidade em 20 mil features. Todos os pontos ficam equidistantes. Nunca use KNN em espaço TF-IDF."*

**Resultados do Tier 1 (2,5 min):**
- Números do LogReg: F1 macro 0,773, recall@crítico 0,500, EACE R$ 977M.
- *"0,50 de recall@crítico significa que metade dos incidentes críticos reais passa despercebida. A classificação manual humana tem ~0,72 — ou seja, o ML clássico ainda perde para o humano em recall crítico. Mas o EACE total já é menor por causa das outras classes."*
- Mostre o threshold tuning: *"Reduzir o threshold de 0,50 para 0,35 na classe crítico aumenta o recall ao custo de precision. A fórmula do threshold ótimo está no business case — é derivada diretamente da matriz de custos."*

**Frase de transição:**
> "O ML clássico nos dá uma base sólida, mas não captura contexto semântico — negação, hipotéticos, linguagem evasiva. Para isso, precisamos de arquitetura sequencial. Entra o spaCy."

---

## Bloco 4 — spaCy: BOW Raso e tok2vec Profundo `[20:00 → 28:00]` ⏱ 8 min

**Objetivo:** mostrar o ganho real do modelo neural, os trade-offs, e por que contexto importa neste domínio.

### O que falar

**spaCy BOW — o trade-off central (2 min):**
- Mostre a tabela comparativa do `04_METRICS_SPACY.md` (seção 1).
- *"O BOW ganha +5,6pp de recall@crítico mas o EACE piora em R$ 51M. Como? Precision@crítico caiu de 0,68 para 0,38 — 61% dos alertas de crítico são falsos positivos. O EACE penaliza isso."*
- Abra `reports/figures/spacy/threshold_tradeoff.png`.
- *"Threshold de 0,10 é extremamente agressivo. Qualquer documento com score de crítico acima de 10% vira um alerta. Isso é necessário porque o modelo BOW raramente atinge scores altos para a classe rara."*

**spaCy tok2vec — a diferença do contexto (2,5 min):**
- Mostre a comparação BOW vs. tok2vec do `06_METRICS_SPACY_DEEP.md` (seção 1).
- Ganhos: +4,2pp recall, +7,6pp F1@crítico, EACE R$ 7,6M melhor.
- *"O tok2vec usa uma CNN de janela de ±4 tokens. Para 'sem H₂S detectado', o encoder vê 'sem' como contexto de 'H₂S' e produz um vetor diferente de 'H₂S detectado'. O BOW não faz essa distinção — os dois documentos têm o mesmo vetor."*
- Abra `reports/figures/spacy_deep/bow_vs_tok2vec.png`.

**Curva de aprendizado — a lição de dataset (2 min):**
- Abra `reports/figures/spacy/learning_curve.png`.
- Destaque o pico em 60%: *"Com 60% dos dados, o BOW atinge recall@crítico = 0,61 — melhor que com 100%. Os últimos 40% têm mais casos ambíguos e mislabeled que confundem o boundary da classe crítico."*
- *"Isso nos diz que anotar mais exemplos críticos especificamente tem impacto muito maior do que aumentar o volume total. Qualidade de anotação > quantidade."*

**Limitação do spaCy (1,5 min):**
- *"Os modelos spaCy operam só no texto — não têm acesso a fator_risco, area_fpso, passagem_turno. Por isso o F1 macro do tok2vec (0,721) é inferior ao ML clássico (0,773) mesmo com recall@crítico maior."*
- *"A pergunta natural é: e se combinarmos os dois?"*

**Frase de transição:**
> "A ideia de combinar o ML clássico — com suas features estruturadas — com o spaCy — com seu contexto semântico — nos leva ao modelo híbrido."

---

## Bloco 5 — Modelos Híbridos e Benchmark Global `[28:00 → 36:00]` ⏱ 8 min

**Objetivo:** mostrar o benchmark completo, as três estratégias de fusão, o vencedor e por que o resultado não é trivial.

### O que falar

**As três estratégias de fusão (2 min):**
- Explique Override, Weighted e Stack em uma frase cada:
  - *"Override: se o spaCy detecta vocabulário crítico com confiança ≥ 0,40, força a predição para crítico. ML clássico decide o resto."*
  - *"Weighted: combinação linear com 40% ML + 60% spaCy. Parece simples — mas depende de calibração similar entre os modelos."*
  - *"Stack: meta-modelo que aprende os pesos de combinação. Mais poderoso em teoria — mais frágil na prática."*
- *"O Weighted com spaCy BOW é o pior de todos — descalibração dos scores BOW destrói a combinação linear. Trocar BOW por tok2vec resolve: +11pp de recall com a mesma fórmula."*

**Benchmark completo — o gráfico mais importante (3 min):**
- Abra `reports/figures/hybrid_full/recall_f1_scatter.png`.
- *"Cada ponto é um sistema. O ideal fica no canto superior direito — alto recall E alta qualidade geral. Nenhum sistema domina sozinho."*
- Destaque os três grupos:
  - **triple_override_deep**: melhor EACE (R$ 951M), bom recall (0,569).
  - **triple_weighted_avg**: melhor F1 macro (0,802), melhor precision@crítico (0,816), recall mediano.
  - **spacy_tok2vec**: melhor recall@crítico (0,597), mas EACE R$ 69M pior que o vencedor.
- *"A fronteira de Pareto inclui: ML clássico, triple_override_deep, spacy_tok2vec. Os demais são dominados — há sempre um sistema melhor em ambas as dimensões."*

**O paradoxo do ranking (2 min):**
- Abra `reports/figures/risk_analysis/violin_eace_mc.png`.
- *"Monte Carlo com 250 mil simulações inverte o ranking. O triple_weighted_avg, que era 4º pelo EACE determinístico, passa a 1º sob incerteza. Por quê?"*
- *"Sua precision@crítico de 0,816 — a mais alta do benchmark — protege o EACE quando os custos de falsos positivos amostram valores altos. Alta precision é robustez a incerteza."*
- Mostre a dominância estocástica: *"A CDF do triple_weighted_avg está sempre à esquerda de todos os outros. Independentemente da aversão ao risco, ele é preferível."*

**A tabela de decisão operacional (1 min):**
- Use a tabela do `08_METRICS_HYBRID_FULL.md` (seção 5):

| Objetivo | Sistema |
|----------|---------|
| Minimizar custo anual | triple_override_deep |
| Minimizar alarmes falsos | triple_weighted_avg |
| Maximizar captura de críticos | spacy_tok2vec |
| Deploy simples | ml_classico |

**Frase de transição:**
> "Mas há 1.024 registros sem rótulo algum. O sistema que construímos pode recuperar esses dados de forma segura?"

---

## Bloco 6 — Recuperação de Anotações e Fechamento `[36:00 → 40:00]` ⏱ 4 min

**Objetivo:** fechar com o impacto prático concreto e a mensagem central que o público vai levar para casa.

### O que falar

**Recuperação controlada por CEE (2 min):**
- *"Para cada registro não anotado, calculamos o Custo Esperado de Erro. Se CEE ≤ R$ 50k, rotulamos automaticamente. Acima disso, vai para revisão humana."*
- *"Por que R$ 50k? É o custo de confundir baixo com crítico — o menor dos erros perigosos. Qualquer predição que pudesse gerar uma confusão crítico→baixo automaticamente excede esse threshold."*
- Mostre a tabela de resultados do `09_UNANNOTATED_RECOVERY.md` (seção 2).
- Destaque: hibrido_override rotula 10% como crítico — **93 registros potencialmente críticos sem rótulo, agora sinalizados para revisão prioritária**.
- *"O backlog não é um problema de qualidade de dados. É uma exposição ao risco de R$ 88 milhões/ano."*

**A mensagem central (2 min):**

Diga com pausa depois de cada linha:

> *"O que aprendemos hoje:"*
>
> *"1. Acurácia mede o modelo. EACE mede o risco do negócio. São coisas diferentes."*
>
> *"2. +1pp de recall@crítico = R$ 33 milhões em custo esperado evitado, por ano, por plataforma."*
>
> *"3. Nenhum modelo vence em tudo. A escolha depende do contexto operacional — latência, tolerância a alarmes falsos, necessidade de explicabilidade."*
>
> *"4. Dados não anotados não são descartáveis. Com threshold de confiança correto, são o próximo passo de melhoria do modelo — sem precisar de LLM."*
>
> *"5. Monte Carlo é a diferença entre saber o que o modelo faz nos dados que você tem — e entender o risco que você assume no mundo real."*

Termine com:
> *"O pipeline completo está no repositório. Cada decisão está documentada. Se você trabalha com segurança industrial, processos críticos ou qualquer domínio onde o custo do erro é assimétrico — esses princípios se aplicam diretamente."*

---

## Resumo dos Tempos

| Bloco | Tema | Tempo | Acumulado |
|-------|------|-------|-----------|
| 1 | O Problema de Negócio | 5 min | 5 min |
| 2 | EDA e Qualidade dos Dados | 7 min | 12 min |
| 3 | Features e ML Clássico | 8 min | 20 min |
| 4 | spaCy BOW e tok2vec | 8 min | 28 min |
| 5 | Híbridos e Benchmark Global | 8 min | 36 min |
| 6 | Recuperação e Fechamento | 4 min | 40 min |

---

## Gráficos Obrigatórios (por ordem de aparição)

1. Tabela de custos — `00_BUSINESS_CASE.md` seção 2.1
2. Tabela EACE por nível de modelo — `00_BUSINESS_CASE.md` seção 3.2
3. `reports/figures/eda/class_distribution.png`
4. `reports/figures/eda/annotation_breakdown.png`
5. `reports/figures/eda/top_tokens_by_class.png`
6. `reports/figures/eda/association_heatmap.png`
7. `reports/figures/features/temporal_features.png`
8. `reports/figures/features/tfidf_projection.png`
9. Tabela ranking EACE — `03_MODEL_SELECTION.md` seção 2
10. `reports/figures/spacy/threshold_tradeoff.png`
11. `reports/figures/spacy_deep/bow_vs_tok2vec.png`
12. `reports/figures/spacy/learning_curve.png`
13. `reports/figures/hybrid_full/recall_f1_scatter.png`
14. `reports/figures/risk_analysis/violin_eace_mc.png`
15. Tabela de decisão operacional — `08_METRICS_HYBRID_FULL.md` seção 5

---

## O que cortar se atrasar

Se perceber que está com 2+ minutos de atraso no bloco 4 (spaCy):

- **Corte no bloco 4:** elimine a discussão da curva de aprendizado (2 min). Vá direto para o resultado do tok2vec e a transição para híbridos.
- **Corte no bloco 5:** mostre apenas o scatter recall × F1 e a tabela de decisão. Elimine o paradoxo do Monte Carlo.
- **Nunca corte:** o número do CFO (bloco 1) e a mensagem central do fechamento (bloco 6). São o início e o fim da narrativa.

---

## Perguntas prováveis da audiência

**"Por que não usar LLM para tudo?"**
> "Custo. 10% dos registros via LLM custa ~R$ 4.500/ano. Escalar para 100% seria R$ 45.000/ano — e a latência de 800ms/registro não serve para alertas em tempo real. A cascata usa LLM só onde nenhum modelo clássico tem confiança suficiente."

**"Como saber quando o modelo degradou em produção?"**
> "Monitorar o drift do vocabulário com janela deslizante de 90 dias. Se novos termos técnicos aparecem com frequência alta e o modelo atribui scores baixos de crítico a registros que operadores classificam como críticos, é sinal de retreinamento necessário."

**"O dataset sintético representa a realidade?"**
> "A distribuição de classes, o padrão de passagem de turno e a sobreposição léxica foram calibrados com base em dados reais offshore. As métricas absolutas (F1 = 0,77) podem ser diferentes em dados reais — mas os rankings relativos entre modelos e a lógica de custo são transferíveis."

**"Como tratar o mislabeling em produção?"**
> "Ciclo de feedback com revisão amostral: 5% das predições de crítico são revisadas por técnico SMS mensalmente. Exemplos corrigidos entram no treino como dados limpos. A taxa de erro de anotação tende a cair abaixo de 3% em 2–3 ciclos."
