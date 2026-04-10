"""
src/quality — Módulo de validação de qualidade de dados.

Hierarquia:
    base.py                  → ABCs: QualityValidator, ExpectationResolver, QualityReportWriterBase
    expectation_resolver.py  → GeExpectationResolver (implementação concreta para GE)
    ge_validator.py          → GreatExpectationsValidator (implementação concreta)
    report_writer.py         → QualityReportWriter (implementação concreta)

Uso via PipelineContext (recomendado):
    context = PipelineContext.from_notebook(__file__)
    context.run_step("quality")

Uso direto (para testes ou integração customizada):
    from src.quality import GreatExpectationsValidator, GeExpectationResolver, QualityReportWriter
"""
from src.quality.base import (
    ExpectationResolver,
    QualityReportWriterBase,
    QualityValidator,
)
from src.quality.expectation_resolver import GeExpectationResolver
from src.quality.ge_validator import GreatExpectationsValidator
from src.quality.report_writer import QualityReportWriter

__all__ = [
    "QualityValidator",
    "ExpectationResolver",
    "QualityReportWriterBase",
    "GeExpectationResolver",
    "GreatExpectationsValidator",
    "QualityReportWriter",
]
