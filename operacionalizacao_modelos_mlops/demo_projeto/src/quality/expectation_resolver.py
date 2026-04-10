"""
src/quality/expectation_resolver.py — Resolução dinâmica de classes GE a partir do YAML.

Responsabilidade única: traduzir um nome de expectation (string do YAML) para
a classe Python correspondente no módulo great_expectations.expectations.

Aceita snake_case (ex.: "expect_column_values_to_not_be_null") e PascalCase
(ex.: "ExpectColumnValuesToNotBeNull") — ambas as convenções são válidas no YAML.
"""
from __future__ import annotations

from src.quality.base import ExpectationResolver


class GeExpectationResolver(ExpectationResolver):
    """
    Resolve nomes de expectation para classes concretas do Great Expectations.

    Recebe o módulo `great_expectations.expectations` (gxe) no construtor
    para facilitar testes — em testes unitários podemos injetar um módulo
    falso (mock) sem instalar o GE de verdade.

    Args:
        gxe: Módulo great_expectations.expectations já importado.
    """

    def __init__(self, gxe) -> None:
        self._gxe = gxe

    # ── API pública ────────────────────────────────────────────────────────────

    def resolve(self, type_name: str) -> type:
        """
        Resolve o nome de uma expectation para sua classe concreta.

        Tenta PascalCase primeiro (convenção nativa do GE), depois tenta o
        nome exatamente como fornecido.

        Args:
            type_name: Nome em snake_case ou PascalCase.

        Returns:
            Classe da expectation pronta para ser instanciada.

        Raises:
            AttributeError: se o nome não existir no módulo gxe.
        """
        pascal = self._para_pascal(type_name)

        if hasattr(self._gxe, pascal):
            return getattr(self._gxe, pascal)

        if hasattr(self._gxe, type_name):
            return getattr(self._gxe, type_name)

        raise AttributeError(
            f"Expectation '{type_name}' não encontrada no módulo great_expectations.expectations.\n"
            f"Verifique o nome no quality.yaml — use snake_case ou PascalCase exato."
        )

    # ── Helpers privados ───────────────────────────────────────────────────────

    @staticmethod
    def _para_pascal(snake: str) -> str:
        """Converte snake_case para PascalCase (ex.: row_count → RowCount)."""
        return "".join(palavra.capitalize() for palavra in snake.split("_"))
