"""
src/quality/report_writer.py — Serialização do relatório de qualidade em JSON.

Responsabilidade única: transformar o summary dict retornado pelo validator
em um arquivo JSON legível e datado.

O objeto `results` do GE não é diretamente JSON-serializable; este módulo
extrai e normaliza apenas as informações relevantes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.quality.base import QualityReportWriterBase


class QualityReportWriter(QualityReportWriterBase):
    """
    Persiste o resumo de qualidade em um arquivo JSON datado.

    Args:
        logger: Logger compartilhado do pipeline.
    """

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger)

    # ── API pública ────────────────────────────────────────────────────────────

    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        """
        Serializa o resumo em JSON e salva em output_dir.

        O nome do arquivo inclui um timestamp UTC para permitir múltiplas
        execuções sem sobrescrever relatórios anteriores.

        Args:
            summary:    Retorno de GreatExpectationsValidator.validate().
            output_dir: Diretório de destino (criado automaticamente se ausente).

        Returns:
            Path absoluto do arquivo JSON gerado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"quality_report_{timestamp}.json"

        report = {
            "success": summary["success"],
            "total":   summary["total"],
            "passed":  summary["passed"],
            "failed":  summary["failed"],
            "details": self._extrair_detalhes(summary["results"]),
        }

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

        self._logger.info("Relatório de qualidade salvo: %s", report_path)
        return report_path

    # ── Helpers privados ───────────────────────────────────────────────────────

    @staticmethod
    def _extrair_detalhes(results) -> list[dict[str, Any]]:
        """
        Extrai informações serializáveis de cada resultado GE.

        O objeto result do GE pode conter listas longas (ex.: valores inesperados).
        Filtramos valores muito grandes para manter o JSON legível.
        """
        detalhes = []
        for r in results.results:
            kwargs_brutos = dict(r.expectation_config.kwargs)
            coluna        = kwargs_brutos.pop("column", None)
            resultado_raw = r.result if isinstance(r.result, dict) else {}

            detalhes.append({
                "type":    r.expectation_config.type,
                "column":  coluna,
                "success": r.success,
                "kwargs":  kwargs_brutos,
                "result":  {
                    k: v for k, v in resultado_raw.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 500
                },
            })
        return detalhes
