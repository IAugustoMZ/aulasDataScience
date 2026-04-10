"""
test_parquet_writer.py — Testes unitários para CsvToParquetIngester.

Estratégia:
  - Todos os testes usam arquivos reais em diretórios pytest tmp_path (sem mocks
    necessários para a lógica principal — o PyArrow roda localmente).
  - A API do Kaggle nunca é acessada.
  - Cada teste cobre um único comportamento de forma isolada.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parquet_writer import CsvToParquetIngester


# ── Helpers ───────────────────────────────────────────────────────────────────

def _criar_ingester(
    raw_dir: Path,
    output_path: Path,
    null_logger,
    required_columns: list[str] | None = None,
    validate_schema: bool = True,
    skip_if_exists: bool = False,
    force: bool = False,
    compression: str = "snappy",
) -> CsvToParquetIngester:
    """Cria um CsvToParquetIngester com parâmetros padrão para os testes."""
    return CsvToParquetIngester(
        raw_dir=raw_dir,
        output_path=output_path,
        compression=compression,
        validate_schema=validate_schema,
        required_columns=required_columns or [],
        skip_if_exists=skip_if_exists,
        force=force,
        logger=null_logger,
    )


# ── Caminho feliz ─────────────────────────────────────────────────────────────

class TestCaminhoFeliz:
    def test_converte_csv_para_parquet(self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger, required_columns):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger, required_columns)

        resultado = ingester.run()

        assert resultado == saida
        assert saida.exists()
        assert saida.stat().st_size > 0

    def test_saida_tem_numero_correto_de_linhas(self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger)
        ingester.run()

        df = pd.read_parquet(saida)
        assert len(df) == 3

    def test_saida_tem_colunas_corretas(self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger, required_columns):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger, required_columns)
        ingester.run()

        schema = pq.read_schema(str(saida))
        for col in required_columns:
            assert col in schema.names

    def test_concatena_multiplos_arquivos_csv(self, tmp_raw_dir, tmp_processed_dir, null_logger):
        """Quando múltiplos CSVs estão presentes, devem ser concatenados."""
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(tmp_raw_dir / "parte1.csv", index=False)
        pd.DataFrame({"a": [5, 6], "b": [7, 8]}).to_csv(tmp_raw_dir / "parte2.csv", index=False)

        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger)
        ingester.run()

        df = pd.read_parquet(saida)
        assert len(df) == 4


# ── Idempotência / lógica de skip ─────────────────────────────────────────────

class TestLogicaSkip:
    def test_pula_quando_saida_existe_e_skip_habilitado(
        self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger
    ):
        saida = tmp_processed_dir / "out.parquet"

        # Primeira execução: cria o arquivo
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger, skip_if_exists=False)
        ingester.run()
        mtime_primeira = saida.stat().st_mtime

        # Segunda execução: deve pular e não sobrescrever
        ingester2 = _criar_ingester(tmp_raw_dir, saida, null_logger, skip_if_exists=True)
        ingester2.run()
        mtime_segunda = saida.stat().st_mtime

        assert mtime_primeira == mtime_segunda

    def test_force_sobrescreve_saida_existente(
        self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger
    ):
        saida = tmp_processed_dir / "out.parquet"

        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger, skip_if_exists=False)
        ingester.run()
        mtime_primeira = saida.stat().st_mtime

        ingester2 = _criar_ingester(
            tmp_raw_dir, saida, null_logger, skip_if_exists=True, force=True
        )
        ingester2.run()
        mtime_segunda = saida.stat().st_mtime

        # Arquivo foi re-escrito, mtime deve ser >= ao original
        assert mtime_segunda >= mtime_primeira


# ── Validação de schema ───────────────────────────────────────────────────────

class TestValidacaoSchema:
    def test_aprovado_quando_todas_colunas_obrigatorias_presentes(
        self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger, required_columns
    ):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(
            tmp_raw_dir, saida, null_logger,
            required_columns=required_columns,
            validate_schema=True,
        )
        ingester.run()  # não deve levantar exceção

    def test_levanta_quando_coluna_obrigatoria_ausente(
        self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger
    ):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(
            tmp_raw_dir, saida, null_logger,
            required_columns=["coluna_inexistente"],
            validate_schema=True,
        )
        with pytest.raises(ValueError, match="Validação de schema falhou"):
            ingester.run()

    def test_pula_validacao_quando_desabilitada(
        self, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger
    ):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(
            tmp_raw_dir, saida, null_logger,
            required_columns=["coluna_inexistente"],
            validate_schema=False,  # desabilitado — não deve levantar exceção
        )
        ingester.run()


# ── Tratamento de erros ───────────────────────────────────────────────────────

class TestTratamentoErros:
    def test_levanta_quando_sem_csvs_no_diretorio_raw(
        self, tmp_raw_dir, tmp_processed_dir, null_logger
    ):
        saida = tmp_processed_dir / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger)

        with pytest.raises(FileNotFoundError, match="Nenhum arquivo .csv encontrado"):
            ingester.run()

    def test_cria_diretorio_de_saida_se_inexistente(
        self, tmp_raw_dir, tmp_path, sample_csv, null_logger
    ):
        # Diretório de saída ainda não existe
        saida = tmp_path / "aninhado" / "profundo" / "out.parquet"
        ingester = _criar_ingester(tmp_raw_dir, saida, null_logger)
        ingester.run()

        assert saida.exists()


# ── Opções de compressão ──────────────────────────────────────────────────────

class TestCompressao:
    @pytest.mark.parametrize("codec", ["snappy", "gzip", "none"])
    def test_codecs_de_compressao_suportados(
        self, codec, tmp_raw_dir, tmp_processed_dir, sample_csv, null_logger
    ):
        saida = tmp_processed_dir / f"out_{codec}.parquet"
        ingester = _criar_ingester(
            tmp_raw_dir, saida, null_logger, compression=codec
        )
        ingester.run()
        assert saida.exists()
