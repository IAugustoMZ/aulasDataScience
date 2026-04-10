"""
preprocessing/pipeline_builder.py — Construtor do Pipeline de Pré-processamento.

Responsabilidade única: ler a configuração do YAML e montar um sklearn.Pipeline
com os transformadores stateless na ordem correta.

Princípio de design — Separação entre política e mecanismo:
  • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
  • Mecanismo → este arquivo + transformers/ (COMO executar cada transformação)

⚠  Transformadores stateful (GroupMedianImputer, StandardScalerTransformer)
   NÃO são incluídos aqui. Eles devem ser aplicados DENTRO do pipeline de
   modelagem (modelagem.py), APÓS o split treino/holdout, para evitar data leakage.
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import (
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    OceanProximityEncoder,
    FeatureSelector,
)


class PreprocessingPipelineBuilder:
    """
    Constrói um sklearn.Pipeline de feature engineering a partir do config YAML.

    A ordem das etapas é fixa e reflete as dependências entre transformações:
    1. BinaryFlagTransformer     — flags sobre colunas originais (sem dependências)
    2. RatioFeatureTransformer   — razões usam colunas originais
    3. LogTransformer            — aplica log1p (inclui razões recém-criadas)
    4. GeoDistanceTransformer    — usa latitude/longitude originais
    5. PolynomialFeatureTransformer — usa colunas originais (median_income, housing_median_age)
    6. OceanProximityEncoder     — encoding da variável categórica
    7. FeatureSelector           — seleciona o subconjunto final (deve ser o último)

    Uso:
        builder = PreprocessingPipelineBuilder(config=preprocessing_cfg, logger=logger)
        pipeline = builder.build()
        df_transformado = pipeline.fit_transform(df)
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger

    def build(self) -> Pipeline:
        """
        Monta e retorna o sklearn.Pipeline com todas as etapas configuradas.

        Returns:
            sklearn.Pipeline pronto para fit_transform().

        Raises:
            KeyError: Se uma seção obrigatória estiver ausente no config.
        """
        etapas = [
            ("flags_binarias", BinaryFlagTransformer(
                flags=self.config.get("binary_flags", []),
                logger=self.logger,
            )),
            ("razoes", RatioFeatureTransformer(
                ratios=self.config.get("ratio_features", []),
                logger=self.logger,
            )),
            ("log", LogTransformer(
                columns=self.config.get("log_transform", {}).get("columns", []),
                logger=self.logger,
            )),
            ("distancias_geo", GeoDistanceTransformer(
                geo_config=self.config.get("geo_distances", {}),
                logger=self.logger,
            )),
            ("polinomiais", PolynomialFeatureTransformer(
                poly_config=self.config.get("polynomial_features", []),
                logger=self.logger,
            )),
            ("encoding", OceanProximityEncoder(
                enc_config=self.config.get("categorical_encoding", {}),
                logger=self.logger,
            )),
            ("selecao", FeatureSelector(
                features_to_keep=self.config.get("feature_selection", {}).get("features_to_keep", []),
                logger=self.logger,
            )),
        ]

        if self.logger:
            self.logger.info(
                "PreprocessingPipelineBuilder: pipeline montado com %d etapas: %s",
                len(etapas),
                [nome for nome, _ in etapas],
            )

        return Pipeline(etapas)
