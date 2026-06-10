# Caso de Negócio: ML para Classificação de Registros de Segurança em Plataformas FPSO

## Resumo Executivo

Um grande operador de FPSO gera **~15.000 registros de segurança por ano** por unidade de frota.
Hoje, a classificação é manual, inconsistente e 20% incompleta.
Um registro `crítico` não identificado — aquele que deveria ter acionado resposta de emergência mas
não acionou — carrega um custo esperado de **R$ 3,2 milhões** por evento (média ponderada de
multas regulatórias, paradas de produção e custos de investigação).

Este documento quantifica o custo da classificação incorreta para cada tipo de erro,
deriva o **Custo Anual Esperado por Erro (EACE)** para cada nível de ML e demonstra
que a cascata de três níveis reduz o EACE em **~82% em relação à linha de base manual**,
com período de retorno inferior a 5 meses.

---

## 1. O Problema

### 1.1 Classificação manual: lenta, inconsistente e incompleta

| Problema | Impacto medido |
|---|---|
| 12 min/registro × 15.000 registros/ano | **3.000 horas de analista/ano** |
| ~8% de discordância entre anotadores | Registros mal rotulados alimentam estatísticas de risco erradas |
| 20% de backlog (não classificados) | 3.000 registros/ano com **nível de risco desconhecido** |
| Sem ciclo de feedback | Erros se acumulam silenciosamente entre os ciclos de reporte |

**Custo anual da classificação manual:** R$ 360.000/ano  
(3.000h × R$ 120/h — taxa de contratante offshore totalmente carregada)

### 1.2 As quatro classes de risco e suas implicações operacionais

| Classe | Significado operacional | Tempo de resposta esperado | Se não identificada |
|---|---|---|---|
| `baixo` | Manutenção geral, não-conformidade | Manutenção programada | Baixo: consequência mínima |
| `médio` | Quase-acidente, defeito que requer reparo | Ação corretiva em 72h | Moderado: ~R$ 80 mil de custo médio |
| `alto` | Quase-acidente de alta potência, isolamento necessário | Paralisação imediata | Grave: ~R$ 650 mil de custo médio |
| `crítico` | Lesão, incêndio, liberação tóxica, risco estrutural | Resposta de emergência IMEDIATA | Catastrófico: ~R$ 3,2 milhões de custo médio |

---

## 2. O Custo de uma Predição Errada

### 2.1 Matriz de custos por classificação incorreta

A matriz de confusão de um classificador de 4 classes tem 16 células.
Nem todos os erros são equivalentes. A tabela abaixo atribui um **custo de negócio** a cada
tipo de classificação incorreta, derivado de frameworks regulatórios, dados atuariais de seguros
e economia da produção.

> **Como ler a tabela:** Linha = classe verdadeira. Coluna = classe predita.
> Célula = custo estimado (R$) quando um registro com classe verdadeira LINHA é predito como COLUNA.

|  | Pred: baixo | Pred: médio | Pred: alto | Pred: crítico |
|---|---|---|---|---|
| **Real: baixo** | 0 | 2.000 *(falso alarme — inspeção desnecessária)* | 15.000 *(paralisação desnecessária)* | 50.000 *(acionamento completo de resposta de emergência)* |
| **Real: médio** | 80.000 *(ação corretiva perdida → defeito se agrava)* | 0 | 8.000 *(paralisação desnecessária + horas extras)* | 30.000 *(resposta de emergência desnecessária)* |
| **Real: alto** | 650.000 *(isolamento perdido → lesão ou perda de equipamento)* | 120.000 *(resposta atrasada → quase-acidente vira incidente)* | 0 | 25.000 *(acionamento de emergência desnecessário)* |
| **Real: crítico** | **3.200.000** *(emergência perdida → risco de fatalidade / explosão)* | **1.800.000** *(resposta tardia → lesão grave / derramamento maior)* | **400.000** *(resposta parcial → escalada)* | 0 |

**Componentes do custo da célula `crítico → baixo` (R$ 3.200.000):**

| Componente | Valor (R$) | Fonte |
|---|---|---|
| Multa regulatória ANP (média) | 400.000 | Dados de fiscalização ANP 2011–2025 |
| Multa ambiental IBAMA | 800.000 | Proporcional à gravidade do incidente |
| Parada de produção (média de 0,5 dia) | 1.500.000 | Média FPSO R$ 3M/dia × 50% de probabilidade |
| Investigação do incidente + custos jurídicos | 350.000 | Referência setorial |
| Indenizações / seguro trabalhista | 150.000 | Estimativa atuarial |
| **Total** | **3.200.000** | |

### 2.2 Custos de falso positivo (sobreclassificação)

Sobreclassificar um registro `baixo` como `crítico` também tem custo:
- Acionamento da equipe de resposta de emergência: R$ 30.000–80.000 por evento
- Parada de produção durante a investigação: R$ 200.000–600.000
- Fadiga da equipe e dessensibilização a alertas (longo prazo): não quantificável

> **É por isso que Precisão@crítico importa tanto quanto Recall@crítico.**  
> Um modelo que prediz `crítico` para tudo atinge Recall = 1,0, mas  
> paralisará as operações com falsos alarmes em questão de semanas.

---

## 3. Custo Anual Esperado por Erro (EACE) por Nível de Modelo

### 3.1 Como o EACE é calculado

Dado:
- **N = 15.000** registros/ano
- **Distribuição de classes:** baixo 40%, médio 35%, alto 18%, crítico 7%
- **Volume de registros por classe:** baixo 6.000 | médio 5.250 | alto 2.700 | crítico 1.050
- **Matriz de confusão** de cada nível de modelo (estimada a partir de benchmarks; medida nos notebooks 02–05)
- **Matriz de custos** da Seção 2.1

```
EACE = Σ (classe_real, classe_predita) [ N × P(classe_real) × P(classe_predita | classe_real) × Custo(real, predita) ]
```

O termo dominante é sempre `crítico → baixo/médio` devido ao custo assimétrico.

### 3.2 Estimativas de EACE por nível

A tabela abaixo utiliza **taxas de confusão esperadas** derivadas de benchmarks da literatura
para tarefas de NLP em português com 4 classes e desbalanceamento. O Notebook 06 substitui
esses valores pelos medidos no conjunto de teste real.

| Nível | Recall@crítico (est.) | Taxa de erro@crítico | Críticos perdidos/ano (esperado) | EACE — erros em crítico | EACE — todos os erros | vs. Linha de base manual |
|---|---|---|---|---|---|---|
| **Manual (linha de base)** | ~0,72 | ~28% | ~294 | ~R$ 940M | ~R$ 1,05B | — |
| **ML Clássico (TF-IDF + LogReg)** | 0,68 – 0,74 | ~27% | ~284 | ~R$ 909M | ~R$ 985M | –6% |
| **ML Clássico + BERT** | 0,72 – 0,80 | ~22% | ~231 | ~R$ 739M | ~R$ 810M | –23% |
| **spaCy textcat (DL)** | 0,78 – 0,85 | ~17% | ~179 | ~R$ 572M | ~R$ 630M | –40% |
| **LLM few-shot** | 0,82 – 0,88 | ~13% | ~137 | ~R$ 438M | ~R$ 485M | –54% |
| **Cascata (todos os níveis)** | **0,85 – 0,91** | **~10%** | **~105** | **~R$ 336M** | **~R$ 378M** | **–64%** |
| **Cascata + revisão humana** | **0,94 – 0,97** | **~4%** | **~42** | **~R$ 134M** | **~R$ 155M** | **–85%** |

> **Importante:** Estes são *valores esperados* — representam o custo médio de longo prazo  
> caso as mesmas taxas de erro persistam em toda a frota por um ano.  
> Um único evento `crítico → baixo` evitado economiza R$ 3,2 milhões por si só.

**Por que a classificação manual tem desempenho ruim:** anotadores humanos apresentam ~28% de taxa de erro
em registros `crítico` devido a narrativas ambíguas, fadiga e alta carga de trabalho.
A fronteira `crítico ↔ alto` é a mais difícil (ambas exigem ação imediata;
a distinção está em se a resposta é "agora" versus "urgência programada").

### 3.3 O custo por ponto percentual de Recall@crítico

Com 1.050 registros `crítico` por ano:

```
Cada +1pp de Recall@crítico = 10,5 eventos críticos a menos perdidos/ano
Cada evento crítico evitado = ~R$ 3.200.000
Cada +1pp de Recall@crítico ≈ R$ 33.600.000 em custo esperado evitado
```

Este é o número que torna a decisão de investimento óbvia.
Mesmo um modelo que melhore o Recall@crítico em apenas 3pp (ex.: 0,72 → 0,75) economiza
**~R$ 100 milhões** em custo anual esperado em uma frota de 3 FPSOs.

---

## 4. Threshold de Decisão Sensível ao Custo

Classificadores padrão usam um threshold de decisão de 0,50 (prediz a classe com
maior probabilidade softmax). Isso é ótimo para acurácia — não para custo.

### 4.1 Threshold ótimo para recall@crítico

Dada a matriz de custos assimétrica, o threshold ótimo para `crítico` é:

```
threshold_ótimo = C(FN) / (C(FN) + C(FP))

Onde:
  C(FN) = custo de um falso negativo (crítico perdido) = R$ 3.200.000
  C(FP) = custo de um falso positivo (sobreclassificado como crítico) = R$ 50.000

threshold_ótimo = 3.200.000 / (3.200.000 + 50.000) ≈ 0,985
```

Isso significa: **predizer `crítico` mesmo quando o modelo tem apenas ~1,5% de confiança de que não é**.
Na prática, um threshold funcional fica em torno de **0,35–0,45** para a classe `crítico`
(abaixo do padrão 0,50), o que aumenta o recall ao custo de alguma precisão.

> **Isso é demonstrado no Notebook 06:** o gráfico de varredura de threshold mostra como
> reduzir o threshold de decisão para `crítico` troca precisão por recall,
> e o ponto de operação ótimo em custo é identificado para cada nível de modelo.

### 4.2 A curva de tradeoff precisão-recall-custo

Cada nível de modelo produz uma curva de custo diferente conforme o threshold varia:

```
Threshold → 0,0:  Recall@crítico = 1,0, Precisão@crítico → baixa, EACE dominado pelo custo de FP
Threshold → 0,5:  Ponto de operação padrão (ótimo em acurácia)
Threshold → 1,0:  Recall@crítico → 0, todos os críticos perdidos, EACE dominado pelo custo de FN

Threshold ótimo em custo: o mínimo da curva EACE(threshold)
```

A arquitetura em cascata implementa naturalmente uma versão suave disso:
o nível LLM tem um campo explícito de `justificativa` e threshold de confiança mais baixo,
tornando-o mais agressivo na sinalização de `crítico` quando a evidência é ambígua.

---

## 5. Análise de Breakeven: Quando Cada Upgrade se Paga?

### 5.1 Upgrade ML Clássico → spaCy DL

| Item | Valor |
|---|---|
| Custo incremental de treinamento + infraestrutura | ~R$ 40.000/ano |
| Redução esperada de EACE | ~R$ 355.000.000/ano (escala de frota) |
| Breakeven | **< 1 hora de custo esperado evitado** |

O upgrade para DL se paga no instante em que classifica corretamente
um registro `crítico` que o ML Clássico teria perdido.

### 5.2 Upgrade spaCy DL → LLM (few-shot)

| Item | Valor |
|---|---|
| Custo incremental de API LLM (10% dos registros × R$ 0,003) | ~R$ 4.500/ano |
| Redução esperada de EACE | ~R$ 145.000.000/ano |
| Breakeven | **24 minutos de custo esperado evitado** |

### 5.3 Cascata completa vs. ML Clássico isolado

| Item | Valor |
|---|---|
| Custo incremental total da cascata | ~R$ 50.000/ano |
| Redução esperada de EACE vs. ML Clássico | ~R$ 607.000.000/ano |
| Breakeven | **< 3 segundos de custo esperado evitado** |

---

## 6. Framework de Métricas

### 6.1 Métricas primárias de ML (todas medidas no mesmo conjunto de teste)

| Métrica | Fórmula | Mínimo aceitável | Significado de negócio |
|---|---|---|---|
| **F1 Macro** | média(F1 por classe) | ≥ 0,70 | Qualidade balanceada por classe |
| **Recall@crítico** | VP_c / (VP_c + FN_c) | ≥ 0,75 | % de eventos críticos reais capturados |
| **Precisão@crítico** | VP_c / (VP_c + FP_c) | ≥ 0,60 | % de alertas de crítico que são reais |
| **F1@alto** | média harmônica P/R para alto | ≥ 0,68 | Classe de segundo maior impacto |
| **F1 Ponderado** | F1 ponderado pelo suporte | ≥ 0,75 | Throughput em todas as classes |

### 6.2 Métricas de negócio (calculadas no Notebook 06)

| Métrica | Fórmula | Unidade |
|---|---|---|
| **EACE** | Σ taxa_confusão × matriz_custo × volume | R$/ano |
| **Redução de EACE vs. linha de base** | (EACE_base − EACE_modelo) / EACE_base | % |
| **Custo por crítico correto** | custo_inferência / VP_crítico | R$/evento |
| **ROI** | (redução_EACE − custo_sistema) / custo_sistema | x |
| **Taxa de recuperação de anotação** | auto-rotulados / total_não_classificados | % |
| **Custo de inferência por 1.000 registros** | custo_API × volume | R$ |
| **Latência p50 / p95** | medida no conjunto de teste | ms |

### 6.3 Tabela comparativa (preenchida pelos notebooks 03–06)

| Métrica | ML Clássico | Clássico + BERT | spaCy DL | LLM zero-shot | LLM few-shot | Cascata |
|---|---|---|---|---|---|---|
| F1 Macro | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* |
| Recall@crítico | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* |
| Precisão@crítico | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* |
| EACE (R$/ano) | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* |
| Redução de EACE vs. manual | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* | *A definir* |
| Custo por 1k registros (R$) | ~0 | ~0 | ~0 | ~3,00 | ~3,50 | ~0,35 |
| Latência p50 (ms) | ~2 | ~80 | ~30 | ~800 | ~900 | ~15 |
| Explicabilidade | Coef. globais | Parcial | Parcial | Texto justificado | Texto justificado | Em camadas |

### 6.4 Rigor estatístico

Todas as métricas reportadas como **média ± IC 95%** (1.000 reamostras bootstrap no conjunto de teste).  
Comparação entre modelos: teste de McNemar (par a par), Q de Cochran (multi-modelo).  
Uma melhoria é **estatisticamente significativa** se p < 0,05 E Δ F1 macro ≥ 0,02.

---

## 7. Recuperação de Anotações

Para os 10.158 registros não classificados no backlog, a cascata aplica
uma **estratégia de threshold de confiança**:

| Faixa de confiança | Ação | Volume esperado | Contribuição esperada de EACE se ignorado |
|---|---|---|---|
| ≥ 0,90 (níveis concordam) | Auto-rotular — emitir classificação | ~6.100 registros | ~R$ 42M/ano |
| 0,75 – 0,90 | Rótulo suave — revisão expedita pelo analista | ~2.500 registros | ~R$ 28M/ano |
| < 0,75 | Reter — classificação manual completa | ~1.558 registros | ~R$ 18M/ano |

**O backlog não é um problema de qualidade de dados. É uma exposição ao risco de R$ 88 milhões/ano.**

---

## 8. Resumo de ROI

| | Conservador | Esperado | Otimista |
|---|---|---|---|
| Custo anual do sistema | R$ 80.000 | R$ 65.000 | R$ 50.000 |
| Redução de EACE (Cascata vs. Manual) | 60% | 82% | 90% |
| Horas de analista economizadas | 50% | 65% | 75% |
| Custo de analista economizado | R$ 180.000 | R$ 234.000 | R$ 270.000 |
| Custo regulatório / parada evitado | R$ 50.000 | R$ 300.000 | R$ 2.000.000 |
| **Benefício líquido anual** | **R$ 150.000** | **R$ 469.000** | **R$ 2.220.000** |
| **ROI** | **1,9×** | **7,2×** | **44,4×** |
| **Período de retorno** | 6,4 meses | 1,7 meses | < 1 mês |

> O cenário conservador exclui paradas evitadas (assume que nenhum evento crítico  
> é diretamente desencadeado por uma classificação perdida no ano 1).  
> O cenário esperado inclui um evento de escalada evitado por ano.  
> O cenário otimista inclui um dia de parada de produção evitada.

---

## 9. Framework de Decisão

### Qual nível para qual contexto operacional?

| Contexto | Nível recomendado | Justificativa |
|---|---|---|
| Feed de alertas em tempo real (SLA < 10ms) | Apenas Nível 1 | Restrição de latência |
| Processamento em lote > 100k registros | Nível 1 + 2, sem LLM | Restrição de custo |
| Trilha de auditoria regulatória exigida | Nível 1 (LogReg) | Pesos de features globais = explicabilidade |
| Novo tipo de incidente / zero exemplos de treino | LLM few-shot Nível 3 | Generalização sem retreinamento |
| Recuperação de backlog | Cascata completa + thresholds de confiança | Melhor tradeoff acurácia/custo |
| Investigação HIPO (qualquer fatalidade potencial) | Nível 3 + revisão humana | Recall@crítico máximo |
| Custo sem restrição, acurácia é tudo | Cascata + humano no loop | Maior redução de EACE |

### Quando NÃO usar ML

- Quando o texto do registro é muito curto (< 5 palavras) — sinal insuficiente para qualquer modelo
- Quando o domínio muda (nova plataforma, novo processo químico) — retreinar primeiro
- Quando a confiança < 0,60 em todos os níveis — revisão humana obrigatória independentemente da política

---

## 10. O que Cada Notebook Demonstra (e a pergunta de negócio que responde)

| Notebook | Pergunta de negócio | Saída principal |
|---|---|---|
| **01 — EDA e Qualidade de Anotação** | Como são nossos dados? Onde está a lacuna de cobertura? | Distribuição de classes, cobertura de anotação, divergência de vocabulário |
| **02 — Análise de Features** | Quais features carregam sinal preditivo — e em que forma? | Ranking por V de Cramér, separação TF-IDF vs. BERT, especificação do ColumnTransformer |
| **03 — Pipeline ML Clássico** | Qual é nossa linha de base sem infraestrutura? O TF-IDF isolado funciona? | EACE da linha de base, importância de features, varredura de threshold de custo |
| **04 — spaCy DL** | Quanto a consciência de sequência melhora o Recall@crítico? | Redução de EACE vs. linha de base, curva de aprendizado |
| **05 — LLM** | Podemos classificar registros nunca vistos no treino? Qual o custo? | EACE para zero-shot vs. few-shot, custo por crítico correto |
| **06 — Comparação** | Qual nível oferece o melhor ROI? Onde a cascata vence? | Comparação completa de EACE, threshold ótimo em custo, simulação da cascata |

> **A mensagem central deste webinar:**  
> F1 Macro é a nota do modelo. EACE é a conta do negócio.  
> Um modelo que melhora o Recall@crítico de 0,72 para 0,85 tem uma nota melhor  
> — mas, mais importante, economiza ~R$ 429 milhões em custo anual esperado  
> em uma frota de 3 FPSOs. Esse é o número que você leva para o CFO.
