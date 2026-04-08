# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Pré-processamento e Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a TERCEIRA etapa do pipeline de dados.
# Entrada : data/processed/house_price.parquet  ← gerado por qualidade_walkthrough.py
# Saída   : data/features/house_price_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
#   • Mecanismo → src/preprocessing.py       (COMO executar cada transformação)
#
# Para ajustar qualquer transformação (ex: mudar threshold de flag, adicionar
# uma nova feature de razão), edite apenas o YAML. O código não muda.
#
# Transformações cobertas (baseadas no EDA Report):
#   1. Imputação     — total_bedrooms: mediana por ocean_proximity
# ─────────────────────────────────────────────────────────────────────────────
# Configuração do Ambiente
import sys
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq


# Definições de caminhos — mesmo padrão dos outros walkthroughs
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]
prep_yaml_path = CONFIG_DIR / 'preprocessing.yaml'

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml

# ── Importa todos os transformadores do módulo src/preprocessing.py ──────────
from src.preprocessing import (
    GroupMedianImputer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    OceanProximityEncoder,
    FeatureSelector
)

# %%
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
preprocessing = load_yaml(prep_yaml_path)
config.update(preprocessing)  # Mescla as configs (pipeline + preprocessing)

# Configura o logger
log_cfg = config.get('logging')
logger = get_logger(
    name='preprocessamento',
    logging_config=log_cfg
)

logger.info('=== Pré-processamento e Feature Engineering ===')
logger.info('Config carregada: pipeline.yaml + data.yaml + preprocessing.yaml')

# ─────────────────────────────────────────────────────────────────────────────
# Inspeciona a configuração carregada
#
# Ponto de ensino: o aluno vê exatamente o que foi lido do YAML.
# Qualquer mudança no preprocessing.yaml aparece aqui sem alterar o código.
# ─────────────────────────────────────────────────────────────────────────────

# %%
prep_cfg    = config.get('preprocessing', {})
output_dir  = ROOT_DIR / prep_cfg.get('output_dir', 'data/features')
output_path = output_dir / prep_cfg.get('output_filename', 'house_price_features.parquet')
compression = prep_cfg.get('compression', 'snappy')

logger.info('Saída      : %s', output_path)
logger.info('Compressão : %s', compression)

# %%
# Lista as transformações configuradas no YAML
logger.info('Imputações configuradas : %d', len(config.get('imputation', [])))

# ─────────────────────────────────────────────────────────────────────────────
# Inspeciona a configuração carregada
#
# Ponto de ensino: o aluno vê exatamente o que foi lido do YAML.
# Qualquer mudança no preprocessing.yaml aparece aqui sem alterar o código.
# ─────────────────────────────────────────────────────────────────────────────

# %%
prep_cfg    = config.get('preprocessing', {})
output_dir  = ROOT_DIR / prep_cfg.get('output_dir', 'data/features')
output_path = output_dir / prep_cfg.get('output_filename', 'house_price_features.parquet')
compression = prep_cfg.get('compression', 'snappy')

logger.info('Saída      : %s', output_path)
logger.info('Compressão : %s', compression)

# %%
# Lista as transformações configuradas no YAML
logger.info('Imputações configuradas : %d', len(config.get('imputation', [])))
logger.info('Flags binárias          : %d', len(config.get('binary_flags', [])))
logger.info('Features de razão       : %d', len(config.get('ratio_features', [])))
logger.info('Colunas log1p           : %d', len(config.get('log_transform', {}).get('columns', [])))
logger.info('Cidades para distâncias : %d', len(config.get('geo_distances', {}).get('cities', [])))
logger.info('Features polinomiais    : %d', len(config.get('polynomial_features', [])))
logger.info('Configurações de encoding: %s', config.get('categorical_encoding', {}))
logger.info('Features para seleção    : %d', len(config.get('feature_selection', {}).get('features_to_keep', [])))

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Carregamento do Dataset
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Caminho do Parquet gerado pela etapa de ingestão
processed_dir = ROOT_DIR / config['paths']['processed_data_dir']
parquet_path  = processed_dir / config['paths']['output_filename']

logger.info('Lendo: %s', parquet_path)

if not parquet_path.exists():
    raise FileNotFoundError(
        f"Arquivo Parquet não encontrado: {parquet_path}\n"
        "Execute ingestao.py e qualidade.py antes deste script."
    )

# %%
# Inspeciona o schema sem carregar os dados (leitura de metadados é barata)
schema = pq.read_schema(str(parquet_path))
logger.info('Schema original (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-25s %s', field.name, field.type)

# %%
# Carrega o DataFrame completo
df = pq.read_table(str(parquet_path)).to_pandas()
logger.info('Shape original: %s', df.shape)

# %%
# Visão rápida antes das transformações
logger.info(df.head())

# Estatísticas descritivas e nulos — baseline antes do pré-processamento
logger.info('Valores ausentes por coluna:')
for col, n_null in df.isna().sum()[df.isna().sum() > 0].items():
    logger.info('  %-25s %d (%.2f%%)', col, n_null, 100 * n_null / len(df))

# %%
logger.info(df.describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Imputação de Valores Ausentes  [MOVIDA PARA O PIPELINE DE MODELAGEM]
#
# ⚠ AVISO MLOps — Data Leakage
# total_bedrooms tem ~207 NaN (~1%). GroupMedianImputer aprende as medianas
# por grupo (ocean_proximity) no fit() e aplica no transform().
#
# Ajustar o imputador ANTES do split treino/teste usa informação do conjunto
# de teste para preencher NaN no conjunto de treino → data leakage.
#
# total_bedrooms é usado como numerador de bedrooms_per_room (Seção 4).
# Os NaN de total_bedrooms propagam para bedrooms_per_room, que está na
# lista features_to_keep e portanto chega ao modelo.
#
# SOLUÇÃO: GroupMedianImputer é aplicado DENTRO do Pipeline de modelagem
# (modelagem_walkthrough.py), APÓS o split treino/holdout, usando
# ocean_proximity_encoded (ordinal 0-4) como grupo — mesma semântica,
# compatível com o dataset já sem a coluna string original.
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SEÇÃO 2: Imputação — delegada ao Pipeline de modelagem (ver modelagem_walkthrough.py)')
imp_specs = config.get('imputation', [])
logger.info('Especificações (para referência): %s', imp_specs)
n_null = int(df['total_bedrooms'].isna().sum())
logger.info('NaN em total_bedrooms: %d (%.2f%%) — serão imputados no pipeline', n_null, 100 * n_null / len(df))

# %%
# Confirma visualmente: NaN presentes (serão tratados no pipeline)
df[['total_bedrooms', 'ocean_proximity']].describe()

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Flags Binárias
#
# Dois valores têm significado especial neste dataset:
#
# 1. housing_median_age == 52
#    - Teto máximo de coleta — não representa "52 anos", mas "≥ 52 anos"
#    - Blocos com esse valor podem ser sistematicamente mais antigos
#    - A flag age_at_cap permite que o modelo aprenda esse padrão
#
# 2. median_house_value == 500001
#    - Valor censurado: imóvel custou ≥ $500,001 mas foi truncado na coleta
#    - 965 linhas (4.68%) — um modelo padrão subestimará esses imóveis
#    - A flag is_capped_target sinaliza esses pontos para tratamento especial
# ─────────────────────────────────────────────────────────────────────────────

# %%
flags_cfg = config.get('binary_flags', [])
logger.info('─' * 60)
logger.info('SEÇÃO 3: Flags Binárias')

# %%
# Instancia e aplica o transformador de flags (stateless — fit é no-op)
flag_transformer = BinaryFlagTransformer(flags_cfg, logger=logger)
df = flag_transformer.transform(df)

# %%
# Inspeciona os resultados
for spec in flags_cfg:
    new_col = spec['new_column']
    logger.info(
        "'%s': %d linhas = 1 (%.2f%%)",
        new_col,
        int(df[new_col].sum()),
        100 * df[new_col].mean(),
    )

# %%
# Distribuição da variável alvo separada por flag de censura
logger.info(df.groupby('is_capped_target')['median_house_value'].describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Features de Razão
#
# Totais absolutos dependem do tamanho do bloco censitário:
#   - Um bairro com 10.000 cômodos e 2.000 domicílios é diferente de um com
#     100 cômodos e 20 domicílios — mas ambos têm 5 cômodos/domicílio.
#
# EDA mostrou que:
#   bedrooms_per_room       r = -0.256 (vs total_bedrooms r = +0.050)
#   rooms_per_household     r = +0.152 (vs total_rooms    r = +0.134)
#   population_per_household r = -0.247 (vs population    r = -0.025)
#
# Razões são MUITO mais informativas que totais absolutos!
# ─────────────────────────────────────────────────────────────────────────────

# %%
ratio_cfg = config.get('ratio_features', [])
logger.info('─' * 60)
logger.info('SEÇÃO 4: Features de Razão')

# %%
ratio_transformer = RatioFeatureTransformer(ratio_cfg, logger=logger)
df = ratio_transformer.transform(df)

# %%
# Estatísticas das novas features
new_ratio_cols = [spec['name'] for spec in ratio_cfg]
logger.info(df[new_ratio_cols].describe())

# %%
# Correlação com o target — confirma que razões são mais informativas
logger.info('Correlação das razões com median_house_value:')
for col in new_ratio_cols:
    corr = df[col].corr(df['median_house_value'])
    logger.info('  %-30s r = %.3f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — Transformação Logarítmica (log1p)
#
# Features com skewness > 1.5 têm cauda longa que distorce modelos lineares
# e aumenta a influência de outliers extremos.
#
# log1p(x) = log(1+x):
#   - Seguro para x=0 (evita log(0) = -Inf)
#   - Comprime a cauda direita, reduzindo a assimetria
#   - Melhora a linearidade com o target
#
# Colunas originais são mantidas para referência.
# Colunas transformadas recebem prefixo 'log_'.
# ─────────────────────────────────────────────────────────────────────────────

# %%
log_cols = config.get('log_transform', {}).get('columns', [])
logger.info('─' * 60)
logger.info('SEÇÃO 5: Transformação Logarítmica (log1p)')

# %%
log_transformer = LogTransformer(log_cols, logger=logger)
df = log_transformer.transform(df)

# %%
# Comparação de skewness: antes vs depois
logger.info('Comparação de assimetria (skewness):')
for col in log_cols:
    if col in df.columns:
        log_col = f'log_{col}'
        skew_raw = df[col].dropna().skew()
        skew_log = df[log_col].dropna().skew() if log_col in df.columns else float('nan')
        logger.info('  %-30s  raw: %+.2f  → log: %+.2f', col, skew_raw, skew_log)

# %%
# Colunas log criadas
log_created = [f'log_{c}' for c in log_cols if f'log_{c}' in df.columns]
df[log_created].head()

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 6 — Distâncias Geográficas
#
# Latitude e longitude brutas não têm significado direto para modelos lineares.
# Mas a distância ao centro de emprego mais próximo é altamente informativa.
#
# EDA: nearest_city_distance tem r = -0.384 com o target
# (blocos mais próximos de grandes centros valem mais)
#
# Calculamos distância euclidiana em graus para 5 cidades californianas.
# GeoDistanceTransformer adiciona dist_<cidade> para cada cidade +
# nearest_city_distance = mínimo das distâncias.
# ─────────────────────────────────────────────────────────────────────────────

# %%
geo_cfg = config.get('geo_distances', {})
logger.info('─' * 60)
logger.info('SEÇÃO 6: Distâncias Geográficas')
logger.info('Cidades de referência: %s', [c['name'] for c in geo_cfg.get('cities', [])])

# %%
geo_transformer = GeoDistanceTransformer(geo_cfg, logger=logger)
df = geo_transformer.transform(df)

# %%
# Estatísticas das distâncias calculadas
dist_cols = [f"dist_{c['name']}" for c in geo_cfg.get('cities', [])]
nearest_col = geo_cfg.get('nearest_city_column', 'nearest_city_distance')
all_dist_cols = dist_cols + [nearest_col]

logger.info(df[all_dist_cols].describe())

# %%
# Correlação das distâncias com o target
logger.info('Correlação das distâncias com median_house_value:')
for col in all_dist_cols:
    if col in df.columns:
        corr = df[col].corr(df['median_house_value'])
        logger.info('  %-35s r = %.3f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 7 — Features Polinomiais e Interações
#
# A relação entre renda e preço não é perfeitamente linear:
#   - Renda alta tem retorno decrescente sobre o preço → median_income²
#
# Interação income × age captura o efeito de bairros ricos E antigos:
#   - Bairros históricos no Bay Area têm prêmio tanto de renda quanto de idade
#   - EDA: median_income_x_housing_median_age tem r = +0.589 com o target!
#   - Esse é um dos features engineered mais preditivos do dataset.
# ─────────────────────────────────────────────────────────────────────────────

# %%
poly_cfg = config.get('polynomial_features', [])
logger.info('─' * 60)
logger.info('SEÇÃO 7: Features Polinomiais e Interações')

# %%
poly_transformer = PolynomialFeatureTransformer(poly_cfg, logger=logger)
df = poly_transformer.transform(df)

# %%
# Correlação das features polinomiais com o target
poly_cols = [spec['name'] for spec in poly_cfg]
logger.info('Correlação das features polinomiais com median_house_value:')
for col in poly_cols:
    if col in df.columns:
        corr = df[col].corr(df['median_house_value'])
        logger.info('  %-40s r = %.3f', col, corr)

# %%
logger.info(df[poly_cols].describe())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 8 — Encoding da Variável Categórica
#
# ocean_proximity tem 5 categorias com efeito grande (ANOVA η² = 0.238):
# ISLAND, NEAR BAY, NEAR OCEAN, <1H OCEAN, INLAND
#
# Duas representações:
#   1. Ordinal (ocean_proximity_encoded):
#      - 0=ISLAND (mais próximo), 4=INLAND (mais distante)
#      - Útil para GBM/XGBoost, captura a ordem geográfica
#
#   2. One-hot (op_INLAND, op_NEAR BAY, etc.):
#      - Necessário para regressão linear (sem assumir ordem)
#      - Mantemos TODAS as categorias (drop_first=False) para interpretabilidade
#      - op_INLAND é a dummy mais preditiva: r = -0.485 com o target
# ─────────────────────────────────────────────────────────────────────────────

# %%
enc_cfg = config.get('categorical_encoding', {})
logger.info('─' * 60)
logger.info('SEÇÃO 8: Encoding de ocean_proximity')
logger.info('Mapa ordinal: %s', enc_cfg.get('ordinal_map'))

# %%
encoder = OceanProximityEncoder(enc_cfg, logger=logger)
df = encoder.transform(df)

# %%
# Verificação do encoding ordinal
logger.info('Distribuição do encoding ordinal:')
ordinal_col = enc_cfg.get('ordinal_column', 'ocean_proximity_encoded')
logger.info(df[[enc_cfg.get('column', 'ocean_proximity'), ordinal_col]].value_counts().sort_index())

# %%
# Colunas one-hot geradas (prefixo op_)
prefix = enc_cfg.get('one_hot_prefix', 'op')
dummy_cols = [c for c in df.columns if c.startswith(f'{prefix}_')]
logger.info('Dummies criadas: %s', dummy_cols)

# %%
# Correlação das dummies com o target
logger.info('Correlação das dummies com median_house_value:')
for col in dummy_cols:
    corr = df[col].corr(df['median_house_value'])
    logger.info('  %-20s r = %.3f', col, corr)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 9 — Seleção de Features
#
# Após as transformações, o DataFrame tem 50+ colunas.
# Precisamos selecionar o subconjunto recomendado pelo EDA Report:
#
#   ✓ Features de renda, localização, razões, distâncias, interações
#   ✗ Totais brutos (total_rooms, total_bedrooms, population, households)
#     → substituídos pelas razões normalizadas
#
# FeatureSelector emite WARNING (não exceção) para colunas ausentes,
# garantindo que o pipeline continue mesmo que alguma transformação
# anterior tenha sido pulada.
# ─────────────────────────────────────────────────────────────────────────────

# %%
sel_cfg = config.get('feature_selection', {})
features_to_keep = sel_cfg.get('features_to_keep', [])
logger.info('─' * 60)
logger.info('SEÇÃO 9: Seleção de Features')
logger.info('Shape antes da seleção: %s', df.shape)
logger.info('Features solicitadas: %d', len(features_to_keep))

# %%
selector = FeatureSelector(features_to_keep, logger=logger)
df = selector.fit_transform(df)

# %%
logger.info('Shape após seleção: %s', df.shape)
logger.info('Colunas selecionadas:')
for i, col in enumerate(df.columns, 1):
    logger.info('  %2d. %s', i, col)

# %%
logger.info(df.head())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 11 — Persistência do Resultado
#
# Salva o dataset processado em Parquet (formato colunar, comprimido).
# Este arquivo é a ENTRADA da etapa de modelagem.
#
# O relatório pode ser:
#   • Versionado no repositório (rastreabilidade de dados)
#   • Carregado diretamente pelo script de treinamento do modelo
#   • Comparado entre execuções para detectar drift de features
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Cria o diretório de saída se não existir
output_dir.mkdir(parents=True, exist_ok=True)
logger.info('─' * 60)
logger.info('SEÇÃO 11: Persistência')
logger.info('Diretório de saída: %s', output_dir)

# %%
# Salva em Parquet
df.to_parquet(str(output_path), compression=compression, index=False)
size_mb = output_path.stat().st_size / (1024 ** 2)
logger.info('Arquivo salvo: %s (%.2f MB)', output_path, size_mb)

# %%
# Validação pós-escrita: lê o schema sem carregar os dados
schema_out = pq.read_schema(str(output_path))
logger.info('Schema de saída (%d colunas):', len(schema_out))
for field in schema_out:
    logger.info('  %-35s %s', field.name, field.type)

# %%
# Lê uma amostra para confirmação visual
df_check = pd.read_parquet(str(output_path))
logger.info('Verificação pós-leitura — shape: %s', df_check.shape)
logger.info(df_check.head())