"""
ingestion/parquet_writer.py — CsvToParquetIngester

Implementa PipelineStep para a etapa de conversão CSV → Parquet.

Responsabilidades:
  1. Descobrir todos os arquivos CSV em raw_dir (ordem determinística).
  2. Convertê-los em streaming para um único arquivo Parquet via PyArrow (seguro para memória).
  3. Validar opcionalmente que as colunas obrigatórias estão presentes no schema de saída.
  4. Respeitar os flags skip_if_exists / force para idempotência.
"""
from __future__ import annotations

import time
from pathlib import Path
import logging

from src.core.base import PipelineStep


class CsvToParquetIngester(PipelineStep):
    """
    Converte todos os arquivos CSV em raw_dir em um único arquivo Parquet.

    Múltiplos arquivos CSV são concatenados em ordem alfabética; todos devem
    compartilhar o mesmo schema (nomes e tipos de colunas).

    Args:
        raw_dir          : Diretório contendo o(s) arquivo(s) CSV bruto(s).
        output_path      : Caminho completo (incluindo nome do arquivo) para o Parquet de saída.
        compression      : Codec Parquet — snappy | gzip | brotli | none.
        chunk_size_rows  : Linhas por lote de streaming (controla o pico de memória).
        validate_schema  : Se True, verifica se required_columns estão presentes após a escrita.
        required_columns : Nomes de colunas que devem aparecer no schema de saída.
        skip_if_exists   : Se True e a saída já existir, pula a conversão.
        force            : Ignora skip_if_exists — sempre re-ingere.
        logger           : Logger compartilhado do pipeline.
    """

    def __init__(
        self,
        raw_dir: Path,
        output_path: Path,
        compression: str = "snappy",
        chunk_size_rows: int = 50_000,
        validate_schema: bool = True,
        required_columns: list[str] | None = None,
        skip_if_exists: bool = True,
        force: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger or logging.getLogger(__name__))
        self.raw_dir = Path(raw_dir)
        self.output_path = Path(output_path)
        self.compression = compression
        self.chunk_size_rows = chunk_size_rows
        self.validate_schema = validate_schema
        self.required_columns = required_columns or []
        self.skip_if_exists = skip_if_exists
        self.force = force

    # ── Contrato PipelineStep ─────────────────────────────────────────────────

    def run(self) -> Path:
        """
        Executa a conversão CSV → Parquet.

        Returns:
            Path para o arquivo Parquet gerado.

        Raises:
            FileNotFoundError : Se raw_dir não contiver arquivos CSV.
            ValueError        : Se colunas obrigatórias estiverem ausentes após a conversão.
            RuntimeError      : Em caso de falhas de leitura/escrita.
        """
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        import pyarrow.parquet as pq

        if self._deve_pular():
            return self.output_path

        arquivos_csv = sorted(self.raw_dir.glob("*.csv"))
        if not arquivos_csv:
            raise FileNotFoundError(
                f"Nenhum arquivo .csv encontrado em '{self.raw_dir}'. "
                "Execute a etapa de download primeiro."
            )

        self.logger.info(
            "Ingerindo %d arquivo(s) CSV de '%s' -> '%s'",
            len(arquivos_csv),
            self.raw_dir,
            self.output_path,
        )
        for cf in arquivos_csv:
            self.logger.info("  [IN] %s (%.2f MB)", cf.name, cf.stat().st_size / 1024**2)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        opcoes_leitura = pa_csv.ReadOptions(
            block_size=self.chunk_size_rows * 200,
            use_threads=True,
        )
        opcoes_parse = pa_csv.ParseOptions(
            delimiter=",",
            quote_char='"',
            double_quote=True,
            newlines_in_values=False,
        )
        opcoes_conversao = pa_csv.ConvertOptions(
            auto_dict_encode=False,
            include_missing_columns=False,
        )

        inicio = time.monotonic()
        total_linhas = 0
        writer: pq.ParquetWriter | None = None

        try:
            for caminho_csv in arquivos_csv:
                self.logger.info("  [READ] Processando '%s'...", caminho_csv.name)
                linhas_arquivo = 0

                with pa_csv.open_csv(
                    caminho_csv,
                    read_options=opcoes_leitura,
                    parse_options=opcoes_parse,
                    convert_options=opcoes_conversao,
                ) as reader:
                    schema: pa.Schema = reader.schema

                    # Inicializa o writer no primeiro arquivo (o schema determina a saída)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            str(self.output_path),
                            schema=schema,
                            compression=self.compression,
                        )
                        self.logger.info(
                            "Schema: %d coluna(s) - %s", len(schema), schema.names
                        )

                    for batch in reader:
                        writer.write_batch(batch)
                        linhas_arquivo += batch.num_rows
                        total_linhas += batch.num_rows
                        self.logger.debug(
                            "  Lote escrito: %d linhas | total arquivo: %d | total geral: %d",
                            batch.num_rows,
                            linhas_arquivo,
                            total_linhas,
                        )

                self.logger.info("  [OK]   '%s' - %d linhas", caminho_csv.name, linhas_arquivo)
        finally:
            if writer is not None:
                writer.close()

        if self.validate_schema and self.required_columns:
            self._validar_colunas_obrigatorias(pq)

        decorrido = time.monotonic() - inicio
        entrada_mb = sum(f.stat().st_size for f in arquivos_csv) / 1024**2
        saida_mb = self.output_path.stat().st_size / 1024**2
        ratio = entrada_mb / saida_mb if saida_mb > 0 else 0

        self.logger.info(
            "Ingestão concluída em %.1f s | linhas=%d | csv=%.2f MB | parquet=%.2f MB "
            "| ratio=%.2fx | compressão=%s",
            decorrido,
            total_linhas,
            entrada_mb,
            saida_mb,
            ratio,
            self.compression,
        )
        return self.output_path

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _deve_pular(self) -> bool:
        """Retorna True se a saída já existe e o modo de pular está habilitado."""
        if (
            not self.force
            and self.skip_if_exists
            and self.output_path.exists()
            and self.output_path.stat().st_size > 0
        ):
            tamanho_mb = self.output_path.stat().st_size / 1024**2
            self.logger.info(
                "Arquivo já ingerido em '%s' (%.1f MB). Ingestão ignorada. "
                "(Defina force_ingest: true no pipeline.yaml para forçar a re-ingestão.)",
                self.output_path,
                tamanho_mb,
            )
            return True
        return False

    def _validar_colunas_obrigatorias(self, pq) -> None:
        """
        Lê apenas os metadados do rodapé do Parquet para verificar as colunas obrigatórias.

        Não lê nenhuma linha de dados — apenas o schema do rodapé do arquivo.

        Raises:
            ValueError: Se qualquer coluna obrigatória estiver ausente no schema.
        """
        schema = pq.read_schema(str(self.output_path))
        presentes = set(schema.names)
        ausentes = [c for c in self.required_columns if c not in presentes]

        if ausentes:
            raise ValueError(
                f"Validação de schema falhou. Coluna(s) obrigatória(s) ausente(s): {ausentes}\n"
                f"Colunas presentes: {sorted(presentes)}\n"
                "Solução: verifique se o dataset contém as colunas esperadas, "
                "ou atualize 'schema.required_columns' no config/data.yaml."
            )

        self.logger.info(
            "Validação de schema OK — %d coluna(s) obrigatória(s) presentes.",
            len(self.required_columns),
        )
