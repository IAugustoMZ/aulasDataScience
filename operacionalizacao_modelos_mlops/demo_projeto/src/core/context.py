"""
core/context.py — PipelineContext

Objeto central que conecta:
  - Resolução de caminhos  (root, config, raw, processed)
  - Carregamento de config (data.yaml + pipeline.yaml + quality.yaml)
  - Criação do logger      (instância única compartilhada)
  - Despacho de etapas     (context.run_step("ingestion" | "quality"))

Os notebooks só precisam instanciar o PipelineContext e chamar run_step().
Todas as leituras de chaves de config ficam aqui, mantendo os notebooks e
as implementações das etapas livres do conhecimento da estrutura de config.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import logging


class PipelineContext:
    """
    Contexto imutável de execução para uma rodada do pipeline.

    Atributos:
        root_dir      : Raiz absoluta do projeto (pai de notebooks/).
        config_dir    : root_dir / config
        secrets_path  : root_dir / secrets.env
        data_cfg      : Dicionário parseado do data.yaml.
        pipeline_cfg  : Dicionário parseado do pipeline.yaml.
        logger        : Logger compartilhado do pipeline.
        raw_dir       : Caminho absoluto para o diretório de dados brutos.
        processed_dir : Caminho absoluto para o diretório de dados processados.
        output_path   : Caminho completo do arquivo Parquet de saída.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.config_dir = self.root_dir / "config"
        self.secrets_path = self.root_dir / "secrets.env"

        self._garantir_sys_path()

        from src.utils.config_loader import load_yaml
        from src.utils.logger import get_logger

        self.data_cfg: dict[str, Any]     = load_yaml(self.config_dir / "data.yaml")
        self.pipeline_cfg: dict[str, Any] = load_yaml(self.config_dir / "pipeline.yaml")
        self.quality_cfg: dict[str, Any]  = load_yaml(self.config_dir / "quality.yaml")

        self.logger: logging.Logger = get_logger(
            name="pipeline",
            logging_config=self.pipeline_cfg["logging"],
        )

        caminhos = self.pipeline_cfg["paths"]
        self.raw_dir = self.root_dir / caminhos["raw_data_dir"]
        self.processed_dir = self.root_dir / caminhos["processed_data_dir"]
        self.output_path = self.processed_dir / caminhos["output_filename"]

    # ── Helpers de construção ─────────────────────────────────────────────────

    @classmethod
    def from_notebook(cls, notebook_path: str | Path) -> "PipelineContext":
        """
        Constrói o contexto a partir do __file__ de um notebook.

        A raiz do projeto é assumida como o pai do diretório notebooks/.

        Uso (dentro de qualquer notebook):
            context = PipelineContext.from_notebook(__file__)
        """
        root = Path(notebook_path).resolve().parent.parent
        return cls(root)

    def _garantir_sys_path(self) -> None:
        """Adiciona root_dir e config_dir ao sys.path caso ainda não estejam presentes."""
        for p in (str(self.root_dir), str(self.config_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)

    # ── Acessores de config (fonte única da verdade para nomes de chaves) ─────

    @property
    def kaggle_dataset(self) -> str:
        return self.data_cfg["kaggle"]["dataset"]

    @property
    def kaggle_file_pattern(self) -> str:
        return self.data_cfg["kaggle"].get("file_pattern", "*.csv")

    @property
    def kaggle_expected_files(self) -> list[str]:
        return self.data_cfg["kaggle"].get("expected_files") or []

    @property
    def ingest_compression(self) -> str:
        return self.data_cfg["ingest"].get("compression", "snappy")

    @property
    def ingest_chunk_size(self) -> int:
        return self.data_cfg["ingest"].get("chunk_size_rows", 50_000)

    @property
    def ingest_validate_schema(self) -> bool:
        return self.data_cfg["ingest"].get("validate_schema", True)

    @property
    def required_columns(self) -> list[str]:
        return self.data_cfg["schema"].get("required_columns", [])

    @property
    def skip_download(self) -> bool:
        return self.pipeline_cfg["execution"].get("skip_download_if_exists", True)

    @property
    def force_download(self) -> bool:
        return self.pipeline_cfg["execution"].get("force_redownload", False)

    @property
    def skip_ingest(self) -> bool:
        return self.pipeline_cfg["execution"].get("skip_ingest_if_exists", True)

    @property
    def force_ingest(self) -> bool:
        return self.pipeline_cfg["execution"].get("force_ingest", False)

    # ── Despacho de etapas ────────────────────────────────────────────────────

    def run_step(self, etapa: str) -> None:
        """
        Despacha uma etapa nomeada do pipeline.

        Etapas suportadas:
            "ingestion" — verificação de credenciais, download e conversão CSV→Parquet.
            "quality"   — validação de qualidade via Great Expectations.

        Raises:
            ValueError: Se o nome da etapa for desconhecido.
        """
        if etapa == "ingestion":
            self._executar_ingestao()
        elif etapa == "quality":
            self._executar_qualidade()
        else:
            raise ValueError(
                f"Etapa desconhecida: '{etapa}'. "
                f"Etapas suportadas: ['ingestion', 'quality']"
            )

    def _executar_ingestao(self) -> None:
        from src.ingestion.downloader import KaggleDownloader
        from src.ingestion.parquet_writer import CsvToParquetIngester

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        downloader = KaggleDownloader(
            secrets_path=self.secrets_path,
            dataset=self.kaggle_dataset,
            file_pattern=self.kaggle_file_pattern,
            expected_files=self.kaggle_expected_files,
            skip_if_exists=self.skip_download,
            force=self.force_download,
            logger=self.logger,
        )
        downloader.load(destination_dir=self.raw_dir)

        ingester = CsvToParquetIngester(
            raw_dir=self.raw_dir,
            output_path=self.output_path,
            compression=self.ingest_compression,
            chunk_size_rows=self.ingest_chunk_size,
            validate_schema=self.ingest_validate_schema,
            required_columns=self.required_columns,
            skip_if_exists=self.skip_ingest,
            force=self.force_ingest,
            logger=self.logger,
        )
        ingester.run()

    def _executar_qualidade(self) -> None:
        import pandas as pd

        try:
            import great_expectations as gx
            import great_expectations.expectations as gxe
        except ImportError as exc:
            raise ImportError(
                "great-expectations não encontrado.\n"
                "Instale com:  pip install great-expectations>=1.0.0"
            ) from exc

        from src.quality.expectation_resolver import GeExpectationResolver
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.report_writer import QualityReportWriter

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Arquivo Parquet não encontrado: {self.output_path}\n"
                "Execute run_step('ingestion') antes de run_step('quality')."
            )

        self.logger.info("Carregando dados de: %s", self.output_path)
        df = pd.read_parquet(self.output_path)
        self.logger.info("Dados carregados: %d linhas, %d colunas", df.shape[0], df.shape[1])

        # config unificado = pipeline + quality (quality.yaml tem precedência)
        config_unificado = {**self.pipeline_cfg, **self.quality_cfg}

        resolver  = GeExpectationResolver(gxe)
        validator = GreatExpectationsValidator(resolver, self.logger, gx)
        summary   = validator.validate(df, config_unificado)

        quality_cfg = self.quality_cfg.get("quality", {})
        output_dir  = self.root_dir / quality_cfg.get("output_dir", "outputs/quality")

        writer      = QualityReportWriter(self.logger)
        report_path = writer.write(summary, output_dir)

        self.logger.info("-" * 60)
        self.logger.info("Entrada  : %s", self.output_path)
        self.logger.info("Relatório: %s", report_path)
        self.logger.info(
            "Checks   : %d total | %d passaram | %d falharam",
            summary["total"], summary["passed"], summary["failed"],
        )
        self.logger.info("Status   : %s", "APROVADO" if summary["success"] else "REPROVADO")
