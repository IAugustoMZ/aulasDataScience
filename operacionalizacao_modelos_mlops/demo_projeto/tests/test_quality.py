"""
tests/test_quality.py — Testes unitários do módulo src/quality/.

Cobre:
  - GeExpectationResolver     : resolução de nomes válidos e inválidos
  - GreatExpectationsValidator: validação de DataFrame válido e com falhas
  - QualityReportWriter       : criação de diretório, estrutura do JSON gerado
"""
from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def logger_nulo() -> logging.Logger:
    """Logger silencioso para não poluir a saída dos testes."""
    logger = logging.getLogger("test_quality_nulo")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def df_california() -> pd.DataFrame:
    """DataFrame mínimo no estilo California Housing para usar nos testes."""
    return pd.DataFrame({
        "median_house_value": [150_000.0, 250_000.0, 350_000.0],
        "median_income":      [3.5, 5.0, 7.2],
        "housing_median_age": [20, 35, 52],
        "total_rooms":        [800, 1_200, 600],
        "total_bedrooms":     [200, 300, 150],
        "population":         [500, 900, 400],
        "households":         [180, 280, 140],
        "latitude":           [37.88, 37.86, 37.85],
        "longitude":          [-122.23, -122.22, -122.24],
        "ocean_proximity":    ["NEAR BAY", "NEAR BAY", "INLAND"],
    })


@pytest.fixture
def config_qualidade() -> dict[str, Any]:
    """Config mínimo de qualidade compatível com quality.yaml."""
    return {
        "quality": {
            "suite_name": "suite_teste",
            "fail_pipeline_on_error": False,
            "output_dir": "outputs/quality",
        },
        "table_expectations": [
            {
                "type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 10_000},
            }
        ],
        "column_expectations": {
            "median_house_value": [
                {"type": "expect_column_values_to_not_be_null"},
            ]
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# GeExpectationResolver
# ──────────────────────────────────────────────────────────────────────────────

class TestGeExpectationResolver:
    """Testa a resolução de nomes de expectation para classes GE."""

    def _criar_modulo_fake(self, **classes) -> types.ModuleType:
        """Cria um módulo falso com as classes passadas como atributos."""
        modulo = types.ModuleType("gxe_fake")
        for nome, cls in classes.items():
            setattr(modulo, nome, cls)
        return modulo

    def test_resolve_snake_case_converte_para_pascal(self):
        """Nomes em snake_case devem ser convertidos para PascalCase e resolvidos."""
        from src.quality.expectation_resolver import GeExpectationResolver

        classe_fake = type("ExpectTableRowCountToBeBetween", (), {})
        gxe_fake    = self._criar_modulo_fake(ExpectTableRowCountToBeBetween=classe_fake)

        resolver = GeExpectationResolver(gxe_fake)
        resultado = resolver.resolve("expect_table_row_count_to_be_between")

        assert resultado is classe_fake

    def test_resolve_pascal_case_direto(self):
        """Nomes já em PascalCase devem ser resolvidos diretamente."""
        from src.quality.expectation_resolver import GeExpectationResolver

        classe_fake = type("ExpectColumnValuesToNotBeNull", (), {})
        gxe_fake    = self._criar_modulo_fake(ExpectColumnValuesToNotBeNull=classe_fake)

        resolver = GeExpectationResolver(gxe_fake)
        resultado = resolver.resolve("ExpectColumnValuesToNotBeNull")

        assert resultado is classe_fake

    def test_resolve_nome_invalido_levanta_attribute_error(self):
        """Nomes inexistentes no módulo devem levantar AttributeError."""
        from src.quality.expectation_resolver import GeExpectationResolver

        gxe_fake = self._criar_modulo_fake()  # módulo vazio

        resolver = GeExpectationResolver(gxe_fake)

        with pytest.raises(AttributeError, match="não encontrada"):
            resolver.resolve("expect_isso_nao_existe")

    def test_para_pascal_converte_corretamente(self):
        """Método privado _para_pascal deve converter snake_case para PascalCase."""
        from src.quality.expectation_resolver import GeExpectationResolver

        gxe_fake = self._criar_modulo_fake()
        resolver  = GeExpectationResolver(gxe_fake)

        assert resolver._para_pascal("row_count") == "RowCount"
        assert resolver._para_pascal("expect_column_mean_to_be_between") == "ExpectColumnMeanToBeBetween"


# ──────────────────────────────────────────────────────────────────────────────
# GreatExpectationsValidator
# ──────────────────────────────────────────────────────────────────────────────

class TestGreatExpectationsValidator:
    """Testa o validador GE com mocks do contexto efêmero."""

    def _criar_resultado_fake(self, success: bool, tipo: str, coluna: str | None = None):
        """Cria um resultado GE falso compatível com a interface esperada."""
        kwargs_dict = {}
        if coluna:
            kwargs_dict["column"] = coluna

        config = MagicMock()
        config.type = tipo
        config.kwargs = kwargs_dict

        resultado = MagicMock()
        resultado.success = success
        resultado.expectation_config = config
        resultado.result = {"observed_value": 3}
        return resultado

    def _criar_results_fake(self, success: bool, resultados: list):
        """Cria o objeto results retornado pelo GE."""
        results = MagicMock()
        results.success = success
        results.results = resultados
        return results

    def _criar_gx_mock(self, results_fake):
        """Cria um mock do módulo great_expectations."""
        gx = MagicMock()

        # Configura o contexto efêmero encadeado
        context      = MagicMock()
        data_source  = MagicMock()
        asset        = MagicMock()
        batch_def    = MagicMock()
        suite        = MagicMock()
        validation   = MagicMock()

        validation.run.return_value = results_fake

        gx.get_context.return_value                           = context
        context.data_sources.add_pandas.return_value         = data_source
        data_source.add_dataframe_asset.return_value         = asset
        asset.add_batch_definition_whole_dataframe.return_value = batch_def
        context.suites.add.return_value                      = suite
        context.validation_definitions.add.return_value      = validation

        return gx

    def test_valida_dataframe_valido_retorna_success_true(
        self, df_california, config_qualidade, logger_nulo
    ):
        """Um DataFrame que passa em todos os checks deve retornar success=True."""
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.expectation_resolver import GeExpectationResolver

        resultados = [
            self._criar_resultado_fake(True, "expect_table_row_count_to_be_between"),
            self._criar_resultado_fake(True, "expect_column_values_to_not_be_null", "median_house_value"),
        ]
        results_fake = self._criar_results_fake(True, resultados)
        gx_mock      = self._criar_gx_mock(results_fake)

        resolver  = MagicMock(spec=GeExpectationResolver)
        resolver.resolve.return_value = MagicMock(return_value=MagicMock())

        validator = GreatExpectationsValidator(resolver, logger_nulo, gx_mock)
        summary   = validator.validate(df_california, config_qualidade)

        assert summary["success"] is True
        assert summary["total"]   == 2
        assert summary["passed"]  == 2
        assert summary["failed"]  == 0

    def test_detecta_checks_com_falha(self, df_california, config_qualidade, logger_nulo):
        """Checks que falham devem ser contados em 'failed'."""
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.expectation_resolver import GeExpectationResolver

        resultados = [
            self._criar_resultado_fake(True,  "expect_table_row_count_to_be_between"),
            self._criar_resultado_fake(False, "expect_column_values_to_not_be_null", "median_house_value"),
        ]
        results_fake = self._criar_results_fake(False, resultados)
        gx_mock      = self._criar_gx_mock(results_fake)

        resolver  = MagicMock(spec=GeExpectationResolver)
        resolver.resolve.return_value = MagicMock(return_value=MagicMock())

        validator = GreatExpectationsValidator(resolver, logger_nulo, gx_mock)
        summary   = validator.validate(df_california, config_qualidade)

        assert summary["success"] is False
        assert summary["failed"]  == 1
        assert summary["passed"]  == 1

    def test_levanta_runtime_error_quando_fail_on_error_true(
        self, df_california, logger_nulo
    ):
        """Com fail_pipeline_on_error=True e checks falhando, deve levantar RuntimeError."""
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.expectation_resolver import GeExpectationResolver

        config_fail = {
            "quality": {
                "suite_name": "suite_teste",
                "fail_pipeline_on_error": True,
            },
            "table_expectations": [],
            "column_expectations": {},
        }

        resultados   = [self._criar_resultado_fake(False, "expect_table_row_count_to_be_between")]
        results_fake = self._criar_results_fake(False, resultados)
        gx_mock      = self._criar_gx_mock(results_fake)

        resolver  = MagicMock(spec=GeExpectationResolver)
        resolver.resolve.return_value = MagicMock(return_value=MagicMock())

        validator = GreatExpectationsValidator(resolver, logger_nulo, gx_mock)

        with pytest.raises(RuntimeError, match="REPROVADA"):
            validator.validate(df_california, config_fail)

    def test_nao_levanta_erro_quando_fail_on_error_false(
        self, df_california, config_qualidade, logger_nulo
    ):
        """Com fail_pipeline_on_error=False, falhas não devem levantar exceção."""
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.expectation_resolver import GeExpectationResolver

        resultados   = [self._criar_resultado_fake(False, "expect_table_row_count_to_be_between")]
        results_fake = self._criar_results_fake(False, resultados)
        gx_mock      = self._criar_gx_mock(results_fake)

        resolver  = MagicMock(spec=GeExpectationResolver)
        resolver.resolve.return_value = MagicMock(return_value=MagicMock())

        validator = GreatExpectationsValidator(resolver, logger_nulo, gx_mock)
        # fail_pipeline_on_error=False no config_qualidade — não deve levantar
        summary = validator.validate(df_california, config_qualidade)

        assert summary["success"] is False  # falhou, mas não lançou exceção


# ──────────────────────────────────────────────────────────────────────────────
# QualityReportWriter
# ──────────────────────────────────────────────────────────────────────────────

class TestQualityReportWriter:
    """Testa a persistência do relatório de qualidade em JSON."""

    def _criar_resultado_fake(self, success: bool, tipo: str):
        """Cria resultado GE falso para popular o summary."""
        config = MagicMock()
        config.type = tipo
        config.kwargs = {"min_value": 1}

        resultado = MagicMock()
        resultado.success = success
        resultado.expectation_config = config
        resultado.result = {"observed_value": 3}
        return resultado

    def _criar_summary(self, success: bool) -> dict[str, Any]:
        """Cria um summary compatível com o retorno do validator."""
        resultado = self._criar_resultado_fake(success, "expect_table_row_count_to_be_between")
        results   = MagicMock()
        results.results = [resultado]

        return {
            "success": success,
            "total":   1,
            "passed":  1 if success else 0,
            "failed":  0 if success else 1,
            "results": results,
        }

    def test_cria_diretorio_automaticamente(self, tmp_path, logger_nulo):
        """O diretório de saída deve ser criado caso não exista."""
        from src.quality.report_writer import QualityReportWriter

        output_dir = tmp_path / "novo_dir" / "qualidade"
        assert not output_dir.exists()

        writer = QualityReportWriter(logger_nulo)
        writer.write(self._criar_summary(True), output_dir)

        assert output_dir.exists()

    def test_gera_arquivo_json(self, tmp_path, logger_nulo):
        """Deve gerar exatamente um arquivo .json no diretório de saída."""
        from src.quality.report_writer import QualityReportWriter

        writer      = QualityReportWriter(logger_nulo)
        report_path = writer.write(self._criar_summary(True), tmp_path)

        assert report_path.exists()
        assert report_path.suffix == ".json"

    def test_json_contem_campos_obrigatorios(self, tmp_path, logger_nulo):
        """O JSON gerado deve conter os campos: success, total, passed, failed, details."""
        from src.quality.report_writer import QualityReportWriter

        writer      = QualityReportWriter(logger_nulo)
        report_path = writer.write(self._criar_summary(True), tmp_path)

        with open(report_path, encoding="utf-8") as fh:
            dados = json.load(fh)

        for campo in ("success", "total", "passed", "failed", "details"):
            assert campo in dados, f"Campo '{campo}' ausente no relatório JSON"

    def test_nome_arquivo_contem_timestamp(self, tmp_path, logger_nulo):
        """O nome do arquivo deve seguir o padrão quality_report_<timestamp>.json."""
        from src.quality.report_writer import QualityReportWriter

        writer      = QualityReportWriter(logger_nulo)
        report_path = writer.write(self._criar_summary(False), tmp_path)

        assert report_path.name.startswith("quality_report_")

    def test_json_reflete_status_reprovado(self, tmp_path, logger_nulo):
        """Um summary com success=False deve gerar JSON com success=False."""
        from src.quality.report_writer import QualityReportWriter

        writer      = QualityReportWriter(logger_nulo)
        report_path = writer.write(self._criar_summary(False), tmp_path)

        with open(report_path, encoding="utf-8") as fh:
            dados = json.load(fh)

        assert dados["success"] is False
        assert dados["failed"] == 1
