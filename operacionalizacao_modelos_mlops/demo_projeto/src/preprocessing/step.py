"""
preprocessing/step.py — Etapa de Pré-processamento do Pipeline.

Implementa PipelineStep para a etapa de feature engineering:
  Entrada : data/processed/house_price.parquet  (gerado por ingestao + qualidade)
  Saída   : data/features/house_price_features.parquet

Responsabilidades:
  1. Carregar o Parquet de entrada
  2. Construir o pipeline via PreprocessingPipelineBuilder
  3. Aplicar fit_transform (todas as etapas stateless)
  4. Persistir o resultado em Parquet
  5. Validar o schema de saída
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.core.base import PipelineStep
from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from src.utils.config_loader import load_yaml


class PreprocessingStep(PipelineStep):
    """
    Etapa de feature engineering do pipeline MLOps.

    Lê a configuração de preprocessing.yaml, constrói e executa o pipeline de
    transformações stateless, e persiste o dataset final em data/features/.

    Uso (em notebooks/preprocessamento.py):
        ctx  = PipelineContext.from_notebook(__file__)
        step = PreprocessingStep(ctx)
        step.run()
    """

    def __init__(self, context: Any) -> None:
        """
        Args:
            context: PipelineContext — fornece root_dir, config_dir e logger.
        """
        super().__init__(logger=context.logger)
        self.context = context
        self._config = self._carregar_config()

    def _carregar_config(self) -> dict:
        """Mescla pipeline.yaml e preprocessing.yaml em um único dicionário de config."""
        pipeline_cfg = self.context.pipeline_cfg
        prep_cfg = load_yaml(self.context.config_dir / "preprocessing.yaml")
        return {**pipeline_cfg, **prep_cfg}

    @property
    def caminho_entrada(self) -> Path:
        """Caminho do Parquet gerado pela etapa de qualidade."""
        return self.context.output_path

    @property
    def caminho_saida(self) -> Path:
        """Caminho do Parquet de features de saída."""
        prep = self._config.get("preprocessing", {})
        output_dir = self.context.root_dir / prep.get("output_dir", "data/features")
        return output_dir / prep.get("output_filename", "house_price_features.parquet")

    @property
    def compressao(self) -> str:
        """Codec de compressão para o Parquet de saída."""
        return self._config.get("preprocessing", {}).get("compression", "snappy")

    def run(self) -> None:
        """
        Executa a etapa completa de pré-processamento.

        Raises:
            FileNotFoundError: Se o Parquet de entrada não existir.
        """
        self.logger.info("=" * 60)
        self.logger.info("=== Pré-processamento e Feature Engineering ===")
        self.logger.info("Entrada : %s", self.caminho_entrada)
        self.logger.info("Saída   : %s", self.caminho_saida)

        df = self._carregar_dados()
        df_transformado = self._transformar(df)
        self._persistir(df_transformado)

    def _carregar_dados(self) -> pd.DataFrame:
        """Lê o Parquet de entrada e loga o schema e valores ausentes."""
        if not self.caminho_entrada.exists():
            raise FileNotFoundError(
                f"Arquivo Parquet não encontrado: {self.caminho_entrada}\n"
                "Execute ingestao.py e qualidade.py antes deste script."
            )

        schema = pq.read_schema(str(self.caminho_entrada))
        self.logger.info("Schema de entrada (%d colunas):", len(schema))
        for field in schema:
            self.logger.info("  %-25s %s", field.name, field.type)

        df = pq.read_table(str(self.caminho_entrada)).to_pandas()
        self.logger.info("Shape original: %s", df.shape)

        nulos = df.isna().sum()
        nulos = nulos[nulos > 0]
        if not nulos.empty:
            self.logger.info("Valores ausentes por coluna (antes do pré-processamento):")
            for col, n in nulos.items():
                self.logger.info("  %-25s %d (%.2f%%)", col, n, 100 * n / len(df))

        return df

    def _transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constrói e aplica o pipeline de transformações stateless."""
        self.logger.info("-" * 60)
        self.logger.info("Construindo pipeline de transformações...")

        self._logar_resumo_config()

        builder = PreprocessingPipelineBuilder(config=self._config, logger=self.logger)
        pipeline = builder.build()

        self.logger.info("-" * 60)
        self.logger.info("Aplicando transformações...")
        df_out = pipeline.fit_transform(df)

        self.logger.info("-" * 60)
        self.logger.info("Shape após transformações: %s", df_out.shape)
        self.logger.info("Colunas finais (%d):", len(df_out.columns))
        for i, col in enumerate(df_out.columns, 1):
            self.logger.info("  %2d. %s", i, col)

        return df_out

    def _logar_resumo_config(self) -> None:
        """Loga um resumo das transformações configuradas."""
        self.logger.info("Transformações configuradas:")
        self.logger.info("  Flags binárias      : %d", len(self._config.get("binary_flags", [])))
        self.logger.info("  Features de razão   : %d", len(self._config.get("ratio_features", [])))
        self.logger.info(
            "  Colunas log1p       : %d",
            len(self._config.get("log_transform", {}).get("columns", [])),
        )
        self.logger.info(
            "  Cidades (distâncias): %d",
            len(self._config.get("geo_distances", {}).get("cities", [])),
        )
        self.logger.info(
            "  Features polinomiais: %d",
            len(self._config.get("polynomial_features", [])),
        )
        self.logger.info(
            "  Features selecionadas: %d",
            len(self._config.get("feature_selection", {}).get("features_to_keep", [])),
        )

    def _persistir(self, df: pd.DataFrame) -> None:
        """Salva o DataFrame transformado em Parquet e valida o schema de saída."""
        self.caminho_saida.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(str(self.caminho_saida), compression=self.compressao, index=False)
        tamanho_mb = self.caminho_saida.stat().st_size / (1024 ** 2)

        self.logger.info("-" * 60)
        self.logger.info("Arquivo salvo : %s", self.caminho_saida)
        self.logger.info("Tamanho       : %.2f MB", tamanho_mb)
        self.logger.info("Compressão    : %s", self.compressao)

        schema_saida = pq.read_schema(str(self.caminho_saida))
        self.logger.info("Schema de saída (%d colunas):", len(schema_saida))
        for field in schema_saida:
            self.logger.info("  %-35s %s", field.name, field.type)

        df_check = pd.read_parquet(str(self.caminho_saida))
        self.logger.info("Verificação pós-leitura — shape: %s", df_check.shape)
        self.logger.info("=" * 60)
        self.logger.info("Pré-processamento concluído com sucesso.")
