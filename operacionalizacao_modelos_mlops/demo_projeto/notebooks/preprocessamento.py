# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Pré-processamento e Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
#
# TERCEIRA etapa do pipeline de dados.
#   Entrada : data/processed/house_price.parquet  ← gerado por qualidade.py
#   Saída   : data/features/house_price_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e parâmetros)
#   • Mecanismo → src/preprocessing/         (COMO executar cada transformação)
#
# Para ajustar qualquer transformação (threshold de flag, nova feature de razão,
# adicionar colunas ao log1p, etc.), edite apenas o YAML. O código não muda.
#
# Transformações orchestradas por PreprocessingStep (na ordem correta):
#   1. Flags binárias          — age_at_cap (segura para inferência)
#                                is_capped_target criada MAS não incluída em
#                                features_to_keep (target indisponível em inferência)
#   2. Features de razão       — bedrooms_per_room, rooms_per_household, …
#   3. Transformação log1p     — reduz assimetria em colunas skewed
#   4. Distâncias geográficas  — nearest_city_distance (r = -0.384 com target)
#   5. Features polinomiais    — median_income², interação income × age
#   6. Encoding categórico     — ordinal + one-hot de ocean_proximity
#   7. Seleção de features     — subconjunto final definido no YAML
#
# ⚠  Data Leakage — transformadores stateful (GroupMedianImputer,
#    StandardScalerTransformer) NÃO são aplicados aqui.
#    Eles ficam no pipeline de modelagem (modelagem.py), após o split
#    treino/holdout, para evitar contaminação de dados de teste.
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Configura o contexto de execução (caminhos, config, logger)
import sys
from pathlib import Path

# Bootstrap: garante que root_dir esteja no sys.path antes de qualquer import de src/
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

# PipelineContext.from_notebook resolve a raiz do projeto a partir do __file__
# e garante que src/ e config/ estejam no sys.path.
ctx = PipelineContext.from_notebook(__file__)

# %%
# Executa a etapa completa:
#   1. Carrega data/processed/house_price.parquet
#   2. Constrói o sklearn.Pipeline a partir de config/preprocessing.yaml
#   3. Aplica fit_transform (todas as etapas stateless)
#   4. Persiste o resultado em data/features/house_price_features.parquet
#   5. Loga schema, shape, valores ausentes e métricas de saída
step = PreprocessingStep(ctx)
step.run()