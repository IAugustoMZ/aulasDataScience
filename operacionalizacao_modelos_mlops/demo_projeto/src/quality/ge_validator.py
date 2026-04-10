"""
src/quality/ge_validator.py — Implementação concreta do validador usando Great Expectations.

Responsabilidade única: orquestrar a validação GE a partir das regras do YAML.

Esta classe conhece apenas o contrato `ExpectationResolver` (uma ABC) —
nunca instancia o GE diretamente. Isso garante que, se trocarmos de biblioteca,
só esta classe precisa mudar.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.quality.base import ExpectationResolver, QualityValidator


class GreatExpectationsValidator(QualityValidator):
    """
    Valida um DataFrame usando um contexto efêmero do Great Expectations.

    O contexto efêmero não persiste nada em disco, tornando a validação
    reprodutível em qualquer ambiente (CI/CD, notebooks, produção).

    Args:
        resolver: Instância de ExpectationResolver que traduz nomes → classes GE.
        logger:   Logger compartilhado do pipeline.
        gx:       Módulo great_expectations já importado (injetado para facilitar testes).
    """

    def __init__(
        self,
        resolver: ExpectationResolver,
        logger: logging.Logger,
        gx,
    ) -> None:
        super().__init__(logger)
        self._resolver = resolver
        self._gx = gx

    # ── API pública ────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        """
        Executa todas as verificações de qualidade definidas no config.

        Args:
            df:     DataFrame a ser validado (saída da ingestão).
            config: Dicionário unificado com as seções do quality.yaml.

        Returns:
            Dicionário com as chaves:
              success (bool), total (int), passed (int), failed (int), results (Any).

        Raises:
            RuntimeError: se fail_pipeline_on_error=true e algum check falhar.
        """
        quality_cfg         = config.get("quality", {})
        suite_name          = quality_cfg.get("suite_name", "default_suite")
        fail_on_error       = quality_cfg.get("fail_pipeline_on_error", True)
        table_expectations  = config.get("table_expectations", [])
        column_expectations = config.get("column_expectations", {})

        total_definido = len(table_expectations) + sum(
            len(v) for v in column_expectations.values()
        )
        self._logger.info("Iniciando verificações de qualidade: %d checks definidos", total_definido)

        context, batch_def, suite = self._construir_contexto_efemero(df, suite_name)
        self._popular_suite(suite, table_expectations, column_expectations)

        validation_def = context.validation_definitions.add(
            self._gx.ValidationDefinition(
                name=f"{suite_name}_validation",
                data=batch_def,
                suite=suite,
            )
        )

        self._logger.info("Executando validação de qualidade...")
        results = validation_def.run(batch_parameters={"dataframe": df})

        total  = len(results.results)
        passed = sum(1 for r in results.results if r.success)
        failed = total - passed

        self._registrar_resultados(results, total, passed, failed)

        if fail_on_error and not results.success:
            raise RuntimeError(
                f"Qualidade de dados REPROVADA: {failed}/{total} checks falharam.\n"
                "Revise o quality.yaml ou investigue os dados de entrada.\n"
                "Para continuar com falhas, ajuste fail_pipeline_on_error: false"
            )

        return {
            "success": results.success,
            "total":   total,
            "passed":  passed,
            "failed":  failed,
            "results": results,
        }

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _construir_contexto_efemero(self, df: pd.DataFrame, suite_name: str):
        """
        Cria um contexto GE efêmero (sem disco) com datasource pandas.

        Returns:
            Tupla (context, batch_definition, suite).
        """
        context = self._gx.get_context(mode="ephemeral")

        data_source = context.data_sources.add_pandas("pipeline_source")
        asset       = data_source.add_dataframe_asset(name="input_data")
        batch_def   = asset.add_batch_definition_whole_dataframe("full_batch")

        suite = context.suites.add(
            self._gx.ExpectationSuite(name=suite_name)
        )

        return context, batch_def, suite

    def _popular_suite(
        self,
        suite,
        table_expectations: list[dict[str, Any]],
        column_expectations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        Adiciona dinamicamente as expectations à suite a partir das listas do YAML.

        O despacho dinâmico via resolver torna esta classe completamente independente
        de quais expectations específicas o YAML define.
        """
        for exp in (table_expectations or []):
            cls    = self._resolver.resolve(exp["type"])
            kwargs = exp.get("kwargs", {})
            suite.add_expectation(cls(**kwargs))

        for coluna, exps in (column_expectations or {}).items():
            for exp in exps:
                cls    = self._resolver.resolve(exp["type"])
                kwargs = exp.get("kwargs", {})
                suite.add_expectation(cls(column=coluna, **kwargs))

    def _registrar_resultados(self, results, total: int, passed: int, failed: int) -> None:
        """Loga um sumário e o detalhe de cada check."""
        self._logger.info("-" * 60)
        self._logger.info("Resultados da validação de qualidade:")
        self._logger.info("  Total  : %d", total)
        self._logger.info("  Passou : %d", passed)
        self._logger.info("  Falhou : %d", failed)
        self._logger.info("-" * 60)

        for r in results.results:
            status   = "OK  " if r.success else "FAIL"
            exp_type = r.expectation_config.type
            col      = r.expectation_config.kwargs.get("column", "(tabela)")
            self._logger.info("  [%s] %-50s  col=%-25s", status, exp_type, col)
