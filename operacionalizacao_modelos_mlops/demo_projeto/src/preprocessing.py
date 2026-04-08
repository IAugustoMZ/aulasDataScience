"""
preprocessing.py — Transformadores de Pré-processamento e Feature Engineering.

Contexto MLOps:
  Esta é a TERCEIRA etapa do pipeline de dados.
  Entrada : data/processed/house_price.parquet  (gerado por ingestao + qualidade)
  Saída   : data/features/house_price_features.parquet

Princípio de design — Separação entre política e mecanismo:
  • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
  • Mecanismo → este arquivo               (COMO executar cada transformação)

Interface scikit-learn (BaseEstimator + TransformerMixin):
  Cada classe herda de sklearn.base.BaseEstimator e TransformerMixin.

  BaseEstimator  → fornece get_params() e set_params() automaticamente,
                   necessários para GridSearchCV e clone() de Pipeline.
                   REGRA: todos os parâmetros do __init__ devem ter o
                   mesmo nome que o atributo de instância (ex: self.group_col).

  TransformerMixin → fornece fit_transform(X) automaticamente como
                     self.fit(X).transform(X), compatível com sklearn.Pipeline.

  Isso permite compor os transformadores em um Pipeline:
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('imputer',  GroupMedianImputer(...)),
        ('ratios',   RatioFeatureTransformer(...)),
        ('log',      LogTransformer(...)),
        ...
    ])
    pipe.fit_transform(df)

Transformadores stateless (fit é no-op — retorna self sem aprender):
  BinaryFlagTransformer, RatioFeatureTransformer, LogTransformer,
  GeoDistanceTransformer, PolynomialFeatureTransformer, OceanProximityEncoder,
  FeatureSelector

Transformadores stateful (fit aprende parâmetros dos dados de treino):
  GroupMedianImputer        → aprende medianas por grupo
  StandardScalerTransformer → aprende média e desvio padrão por coluna
"""
import numpy as np
import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imputação por Mediana de Grupo
# ─────────────────────────────────────────────────────────────────────────────

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores ausentes usando a mediana do grupo (estratificada).

    Por que mediana por grupo?
    - total_bedrooms tem distribuição assimétrica (skew=3.46) → mediana > média
    - O número de quartos varia sistematicamente por ocean_proximity
    - Imputar com a mediana global ignora essa heterogeneidade regional

    Compatibilidade com sklearn.Pipeline:
        BaseEstimator fornece get_params()/set_params() via introspecção dos
        parâmetros do __init__ (group_col, target_col, logger).
        TransformerMixin fornece fit_transform().

    Exemplo em Pipeline:
        pipe = Pipeline([
            ('imputer', GroupMedianImputer('ocean_proximity', 'total_bedrooms')),
            ('ratios',  RatioFeatureTransformer(ratio_cfg)),
        ])
        pipe.fit_transform(df)

    Atributos aprendidos no fit:
        medians_       (dict): {valor_do_grupo → mediana}
        global_median_ (float): fallback para grupos não vistos no fit
    """

    def __init__(
        self,
        group_col: str,
        target_col: str,
        logger: Any = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        """
        Aprende a mediana de target_col para cada valor de group_col.

        O parâmetro y=None existe por convenção da API scikit-learn —
        não é utilizado (transformador não supervisionado).

        Raises:
            KeyError: Se group_col ou target_col não existirem no DataFrame.
        """
        missing_cols = [c for c in [self.group_col, self.target_col] if c not in X.columns]
        if missing_cols:
            raise KeyError(
                f"GroupMedianImputer.fit: colunas ausentes no DataFrame: {missing_cols}"
            )

        self.medians_ = (
            X.groupby(self.group_col)[self.target_col]
            .median()
            .to_dict()
        )
        self.global_median_ = float(X[self.target_col].median())

        self._log(
            "GroupMedianImputer.fit: medianas aprendidas por '%s' para '%s': %s",
            self.group_col, self.target_col,
            {k: round(v, 1) for k, v in self.medians_.items()},
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Preenche NaN em target_col com a mediana do grupo correspondente.

        Linhas cujo grupo não foi visto no fit recebem a mediana global.
        """
        if not hasattr(self, "medians_"):
            raise RuntimeError(
                "GroupMedianImputer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        n_before = int(X[self.target_col].isna().sum())

        def _fill(row: pd.Series) -> float:
            if pd.isna(row[self.target_col]):
                return self.medians_.get(row[self.group_col], self.global_median_)
            return row[self.target_col]

        X[self.target_col] = X.apply(_fill, axis=1)
        n_after = int(X[self.target_col].isna().sum())

        self._log(
            "GroupMedianImputer.transform: '%s' — NaN antes=%d, depois=%d",
            self.target_col, n_before, n_after,
        )
        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 2. Flags Binárias
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adiciona colunas binárias (0/1) indicando valores capados ou de borda.

    Por que flags binárias?
    - housing_median_age == 52: limite máximo do dataset — não é dado real, é truncamento
    - median_house_value == 500001: valores censurados (verdadeiros ≥ $500k)
      Modelos treinados sem esse flag subestimam imóveis caros.

    Config (preprocessing.yaml → binary_flags):
        - column: "housing_median_age"
          value: 52
          new_column: "age_at_cap"
        - column: "median_house_value"
          value: 500001
          new_column: "is_capped_target"

    Exemplo:
        flags_cfg = config['binary_flags']
        transformer = BinaryFlagTransformer(flags=flags_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, flags: list[dict], logger: Any = None) -> None:
        self.flags = flags
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryFlagTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for spec in self.flags:
            col = spec["column"]
            val = spec["value"]
            new_col = spec["new_column"]

            if col not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "BinaryFlagTransformer: coluna '%s' não encontrada — flag '%s' ignorada.",
                        col, new_col,
                    )
                continue

            X[new_col] = (X[col] == val).astype(int)
            n_flagged = int(X[new_col].sum())
            self._log(
                "BinaryFlagTransformer: '%s' = %s → '%s': %d linhas flagadas (%.2f%%)",
                col, val, new_col, n_flagged, 100 * n_flagged / len(X),
            )
        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 3. Features de Razão
# ─────────────────────────────────────────────────────────────────────────────

class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Cria features de razão (numerador / denominador).

    Por que razões?
    - Totais absolutos (total_rooms, total_bedrooms, population) dependem do
      tamanho do bloco — blocos maiores têm mais tudo.
    - Razões normalizam pelo número de domicílios, tornando features comparáveis
      entre blocos de tamanhos diferentes.
    - EDA: bedrooms_per_room (r=-0.256) supera total_bedrooms (r=+0.050).

    Divisão segura:
    - Denominador zero → NaN (evita divisão por zero)
    - Inf substituído por NaN

    Config (preprocessing.yaml → ratio_features):
        - name: "rooms_per_household"
          numerator: "total_rooms"
          denominator: "households"

    Exemplo:
        ratios_cfg = config['ratio_features']
        transformer = RatioFeatureTransformer(ratios=ratios_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, ratios: list[dict], logger: Any = None) -> None:
        self.ratios = ratios
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "RatioFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.ratios:
            name = spec["name"]
            num = spec["numerator"]
            den = spec["denominator"]

            if num not in X.columns or den not in X.columns:
                if self.logger:
                    self.logger.warning(
                        "RatioFeatureTransformer: colunas '%s' ou '%s' ausentes — '%s' ignorada.",
                        num, den, name,
                    )
                continue

            X[name] = (X[num] / X[den].replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            )
            created.append(name)

        self._log("RatioFeatureTransformer: features criadas: %s", created)
        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformação Logarítmica
# ─────────────────────────────────────────────────────────────────────────────

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica log1p(x) nas colunas especificadas, criando novas colunas log_<nome>.

    Por que log1p e não log?
    - log1p(x) = log(1+x) — seguro para x=0 (sem -Inf)
    - x é clipado em 0 antes da transformação (protege contra negativos)

    Colunas originais são mantidas. Novas colunas têm prefixo 'log_'.
    Exemplo: total_rooms → log_total_rooms

    Config (preprocessing.yaml → log_transform.columns):
        - "total_rooms"
        - "population"
        ...

    Exemplo:
        log_cols = config['log_transform']['columns']
        transformer = LogTransformer(columns=log_cols, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "LogTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []
        skipped: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                skipped.append(col)
                continue

            log_col = f"log_{col}"
            skew_before = float(X[col].dropna().skew())
            X[log_col] = np.log1p(X[col].clip(lower=0))
            skew_after = float(X[log_col].dropna().skew())
            created.append(log_col)

            self._log(
                "LogTransformer: '%s' → '%s'  |  skewness: %.2f → %.2f",
                col, log_col, skew_before, skew_after,
            )

        if skipped and self.logger:
            self.logger.warning(
                "LogTransformer: colunas não encontradas (ignoradas): %s", skipped
            )

        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 5. Distâncias Geográficas
# ─────────────────────────────────────────────────────────────────────────────

class GeoDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Calcula a distância euclidiana (em graus) de cada bloco a cidades de referência.

    Por que distância euclidiana em graus?
    - O dataset usa graus diretamente; distância euclidiana é uma boa
      aproximação local (California abrange ~10° lat × ~10° lon).
    - EDA: nearest_city_distance tem r=-0.384 com o target.

    Colunas criadas:
    - dist_<city_name>  para cada cidade configurada
    - nearest_city_distance  = min de todas as dist_*

    Config (preprocessing.yaml → geo_distances):
        lat_col: "latitude"
        lon_col: "longitude"
        nearest_city_column: "nearest_city_distance"
        cities:
          - name: "san_francisco"
            lat: 37.7749
            lon: -122.4194

    Exemplo:
        geo_cfg = config['geo_distances']
        transformer = GeoDistanceTransformer(geo_config=geo_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, geo_config: dict, logger: Any = None) -> None:
        self.geo_config = geo_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "GeoDistanceTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cities = self.geo_config.get("cities", [])
        lat_col = self.geo_config.get("lat_col", "latitude")
        lon_col = self.geo_config.get("lon_col", "longitude")
        nearest_col = self.geo_config.get("nearest_city_column", "nearest_city_distance")

        if lat_col not in X.columns or lon_col not in X.columns:
            if self.logger:
                self.logger.warning(
                    "GeoDistanceTransformer: colunas '%s'/'%s' ausentes — transformação ignorada.",
                    lat_col, lon_col,
                )
            return X

        X = X.copy()
        dist_cols: list[str] = []

        for city in cities:
            name = city["name"]
            col_name = f"dist_{name}"
            X[col_name] = np.sqrt(
                (X[lat_col] - city["lat"]) ** 2 +
                (X[lon_col] - city["lon"]) ** 2
            )
            dist_cols.append(col_name)

        if dist_cols:
            X[nearest_col] = X[dist_cols].min(axis=1)
            self._log(
                "GeoDistanceTransformer: %d distâncias calculadas: %s | '%s' adicionado.",
                len(dist_cols), dist_cols, nearest_col,
            )

        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 6. Features Polinomiais e Interações
# ─────────────────────────────────────────────────────────────────────────────

class PolynomialFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Cria features quadráticas e termos de interação.

    Suporta dois tipos de especificação:
    - 1 coluna  → quadrado:  name = columns[0]²
    - 2 colunas → interação: name = columns[0] × columns[1]

    Por que features polinomiais?
    - A relação entre renda e preço não é perfeitamente linear
    - median_income_squared captura retorno decrescente em alta renda
    - median_income_x_housing_median_age (r=0.589): bairros ricos E antigos são premium

    Config (preprocessing.yaml → polynomial_features):
        - name: "median_income_squared"
          columns: ["median_income"]
        - name: "median_income_x_housing_median_age"
          columns: ["median_income", "housing_median_age"]

    Exemplo:
        poly_cfg = config['polynomial_features']
        transformer = PolynomialFeatureTransformer(poly_config=poly_cfg, logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, poly_config: list[dict], logger: Any = None) -> None:
        self.poly_config = poly_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "PolynomialFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        created: list[str] = []

        for spec in self.poly_config:
            name = spec["name"]
            cols = spec["columns"]
            missing = [c for c in cols if c not in X.columns]

            if missing:
                if self.logger:
                    self.logger.warning(
                        "PolynomialFeatureTransformer: colunas ausentes %s — '%s' ignorada.",
                        missing, name,
                    )
                continue

            if len(cols) == 1:
                X[name] = X[cols[0]] ** 2
            elif len(cols) == 2:
                X[name] = X[cols[0]] * X[cols[1]]
            else:
                if self.logger:
                    self.logger.warning(
                        "PolynomialFeatureTransformer: '%s' tem %d colunas — apenas 1 ou 2 suportadas.",
                        name, len(cols),
                    )
                continue

            created.append(name)

        self._log("PolynomialFeatureTransformer: features criadas: %s", created)
        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 7. Encoding da Variável Categórica
# ─────────────────────────────────────────────────────────────────────────────

class OceanProximityEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica ocean_proximity em duas representações:

    1. Encoding ordinal (ocean_proximity_encoded):
       Mapa configurado por distância ao oceano: ISLAND=0 ... INLAND=4.
       Útil para modelos baseados em árvore e correlações ordinais.

    2. One-hot dummies (op_INLAND, op_NEAR BAY, etc.):
       Necessárias para regressão linear (sem assumir ordem).
       drop_first=False mantém todas as categorias para máxima transparência.

    Por que dual encoding?
    - ANOVA η²=0.238: ocean_proximity sozinha explica 23.8% da variância
    - Cada categoria tem distribuição de preços distinta
    - Cohen's d (INLAND vs coastal) = -1.24: efeito muito grande

    Config (preprocessing.yaml → categorical_encoding):
        column: "ocean_proximity"
        ordinal_column: "ocean_proximity_encoded"
        ordinal_map: {ISLAND: 0, NEAR BAY: 1, NEAR OCEAN: 2, "<1H OCEAN": 3, INLAND: 4}
        one_hot_prefix: "op"
        drop_first: false

    Exemplo:
        enc_cfg = config['categorical_encoding']
        encoder = OceanProximityEncoder(enc_config=enc_cfg, logger=logger)
        df = encoder.fit_transform(df)
    """

    def __init__(self, enc_config: dict, logger: Any = None) -> None:
        self.enc_config = enc_config
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "OceanProximityEncoder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        column = self.enc_config.get("column", "ocean_proximity")
        ordinal_column = self.enc_config.get("ordinal_column", "ocean_proximity_encoded")
        ordinal_map: dict = self.enc_config.get("ordinal_map", {})
        prefix = self.enc_config.get("one_hot_prefix", "op")
        drop_first: bool = self.enc_config.get("drop_first", False)

        if column not in X.columns:
            if self.logger:
                self.logger.warning(
                    "OceanProximityEncoder: coluna '%s' não encontrada — encoding ignorado.",
                    column,
                )
            return X

        X = X.copy()

        # ── Encoding ordinal ──────────────────────────────────────────────────
        X[ordinal_column] = X[column].map(ordinal_map)
        n_unknown = int(X[ordinal_column].isna().sum())
        if n_unknown > 0 and self.logger:
            self.logger.warning(
                "OceanProximityEncoder: %d linhas com valores de '%s' não mapeados → NaN",
                n_unknown, column,
            )
        self._log(
            "OceanProximityEncoder: ordinal '%s' criado — mapa: %s",
            ordinal_column, ordinal_map,
        )

        # ── One-hot dummies ───────────────────────────────────────────────────
        dummies = pd.get_dummies(
            X[column],
            prefix=prefix,
            drop_first=drop_first,
        ).astype(int)

        X = pd.concat([X, dummies], axis=1)
        self._log(
            "OceanProximityEncoder: dummies criadas: %s", list(dummies.columns)
        )
        return X
    
# ─────────────────────────────────────────────────────────────────────────────
# 8. Seleção de Features
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Seleciona o conjunto final de features para modelagem.

    Por que seleção explícita?
    - Após as transformações, o DataFrame tem 40+ colunas (originais + engineered).
    - Colunas brutas de contagem (total_rooms, etc.) são substituídas por razões.
    - Manter apenas o que vai para o modelo previne vazamento acidental de dados.

    Comportamento tolerante:
    - Colunas ausentes geram WARNING (não exceção) — permite que o pipeline
      continue mesmo que uma transformação anterior tenha sido pulada.
    - Apenas as colunas disponíveis são selecionadas.

    Config (preprocessing.yaml → feature_selection.features_to_keep):
        - "median_income"
        - "bedrooms_per_room"
        - ...

    Exemplo:
        sel_cfg = config['feature_selection']
        selector = FeatureSelector(features_to_keep=sel_cfg['features_to_keep'], logger=logger)
        df = selector.fit_transform(df)
    """

    def __init__(self, features_to_keep: list[str], logger: Any = None) -> None:
        self.features_to_keep = features_to_keep
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        """Valida que todas as features configuradas existem no DataFrame."""
        missing = [c for c in self.features_to_keep if c not in X.columns]
        if missing and self.logger:
            self.logger.warning(
                "FeatureSelector.fit: %d colunas da config ausentes no DataFrame "
                "(serão ignoradas): %s",
                len(missing), missing,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        available = [c for c in self.features_to_keep if c in X.columns]
        dropped = len(self.features_to_keep) - len(available)

        if dropped > 0 and self.logger:
            self.logger.warning(
                "FeatureSelector.transform: %d/%d colunas solicitadas ausentes — ignoradas.",
                dropped, len(self.features_to_keep),
            )

        self._log(
            "FeatureSelector.transform: %d/%d features selecionadas | shape: %s → %s",
            len(available), len(self.features_to_keep),
            X.shape, (len(X), len(available)),
        )
        return X[available].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Escalonamento (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica Z-score normalization: z = (x − μ) / σ.

    Por que StandardScaler?
    - Regressão linear e SVM são sensíveis à escala das features.
    - Gradient boosting e Random Forest NÃO precisam de escalonamento —
      mas é boa prática manter o dataset escalado para comparação entre modelos.
    - Binárias (0/1) e ordinais têm escala interpretável → não escalonar.

    Parâmetros aprendidos no fit (só no dataset de treino!):
        mean_  (dict): {coluna: média}
        std_   (dict): {coluna: desvio padrão}

    Colunas com std=0 são ignoradas (constantes — sem informação).

    Config (preprocessing.yaml → scaling.columns):
        - "median_income"
        - "housing_median_age"
        - ...

    Exemplo:
        scale_cols = config['scaling']['columns']
        scaler = StandardScalerTransformer(columns=scale_cols, logger=logger)
        df = scaler.fit_transform(df)  # em produção: fit no treino, transform no teste
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        """
        Aprende média e desvio padrão das colunas especificadas.

        ATENÇÃO MLOps: sempre chamar fit() apenas no conjunto de treino.
        Usar transform() no conjunto de validação/teste para evitar data leakage.
        """
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        skipped: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                skipped.append(col)
                continue

            mu = float(X[col].mean())
            sigma = float(X[col].std())

            if sigma == 0:
                if self.logger:
                    self.logger.warning(
                        "StandardScalerTransformer.fit: '%s' tem std=0 (constante) — ignorada.",
                        col,
                    )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if skipped and self.logger:
            self.logger.warning(
                "StandardScalerTransformer.fit: colunas ausentes ignoradas: %s", skipped
            )

        self._log(
            "StandardScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Aplica Z-score nas colunas ajustadas no fit.

        Colunas não vistas no fit são mantidas sem alteração.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        scaled: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            scaled.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d colunas escalonadas (z-score).",
            len(scaled),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com média e desvio padrão aprendidos (útil para auditoria)."""
        return pd.DataFrame(
            {"mean": self.mean_, "std": self.std_}
        ).rename_axis("feature")