"""
src/quality.py — Validação de qualidade de dados via Great Expectations.

Design MLOps:
  - Este módulo é AGNÓSTICO ao dataset. Toda a política de qualidade vive
    no arquivo quality.yaml — aqui só existe o mecanismo de execução.
  - Usa contexto efêmero do GE (sem DataContext em disco), ideal para
    pipelines reprodutíveis e ambientes de CI/CD.
  - Suporta expectations de tabela e de coluna definidas via YAML.

Fluxo:
  run_quality_checks(df, config) → summary dict
  save_quality_report(summary, output_dir) → Path do JSON salvo
"""
import json
import pandas as pd
from typing import Any
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger

def _import_ge():
    """Importa great_expectations; levanta ImportError descritivo se ausente."""
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
        return gx, gxe
    except ImportError as exc:
        raise ImportError(
            "great-expectations não encontrado.\n"
            "Instale com:  pip install great-expectations>=1.0.0\n"
            "Ou adicione ao requirements.txt e execute pip install -r requirements.txt"
        ) from exc
    
def _build_ephemeral_context(df: pd.DataFrame, suite_name: str):
    """
    Cria um contexto efêmero GE com datasource pandas e retorna
    (context, batch_definition, suite).

    O contexto efêmero não persiste nada em disco — ideal para pipelines
    reprodutíveis. Toda a configuração vem do YAML, não de arquivos GE.
    """
    gx, _ =_import_ge()

    # contexto efêmero
    context = gx.get_context(
        mode='ephemeral'
    )

    # datasource pandas
    data_source = context.data_sources.add_pandas("pipeline_source")
    asset = data_source.add_dataframe_asset(
        name="input_data"
    )
    batch_def = asset.add_batch_definition_whole_dataframe('full_batch')

    # criar uma suite vazia (expectations serão adicionadas dinamicamente)
    suite = context.suites.add(
        gx.ExpectationSuite(name=suite_name)
    )

    return context, batch_def, suite

def _snake_to_pascal(snake_str: str) -> str:
    """Converte snake_case para PascalCase (ex.: row_count → RowCount)."""
    return ''.join(word.capitalize() for word in snake_str.split('_'))

def _resolve_expectation_class(gxe, type_name: str):
    """
    Resolve a classe GE a partir do nome em snake_case ou PascalCase.

    Aceita ambas as convenções para não forçar o usuário do YAML a memorizar
    a capitalização exata.

    Raises:
        AttributeError: se o tipo não existir no módulo de expectations do GE.
    """
    # tentar PascalCase primeiro
    pascal = _snake_to_pascal(type_name)

    if hasattr(gxe, pascal):
        return getattr(gxe, pascal)
    elif hasattr(gxe, type_name):
        return getattr(gxe, type_name)
    else:
        raise AttributeError(
            f"Expectation type '{type_name}' não encontrado no módulo great_expectations.expectations."
        )

def _populate_suite_with_expectations(
    suite,
    table_expectations: list[dict[str, Any]],
    column_expectations: dict[str, list[dict[str, Any]]],
    gxe
) -> None:
    """
    Adiciona dynamicamente as expectations à suite a partir das listas do YAML.

    Estratégia:
      - table_exps: expectations sem coluna (ex.: row count, schema)
      - column_exps: dict coluna → lista de expectations

    O despacho dinâmico via _resolve_expectation_class torna este módulo
    completamente independente de quais expectations específicas o YAML define.
    """
    # expectations de tabela
    for exp in (table_expectations or []):
        exp_class = _resolve_expectation_class(gxe, exp['type'])
        kwargs = exp.get('kwargs', {})
        suite.add_expectation(
            exp_class(**kwargs)
        )

    # expectations de coluna
    for col, exps in (column_expectations or {}).items():
        for exp in exps:
            exp_class = _resolve_expectation_class(gxe, exp['type'])
            kwargs = exp.get('kwargs', {})
            suite.add_expectation(
                exp_class(column=col, **kwargs)
            )

# ----------------------------------------------------------------------------------------------------------------    
# função principal
def run_quality_checks(
    df: pd.DataFrame,
    config: dict[str, Any],
    logging_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Executa todas as verificações de qualidade definidas no config.

    Args:
        df:             DataFrame a ser validado (saída da ingestão).
        config:         Dicionário combinado (pipeline.yaml + quality.yaml).
        logging_config: Seção 'logging' do pipeline.yaml.

    Returns:
        Dicionário com:
          success  (bool)   — True se todas as expectations passaram
          total    (int)    — número total de checks
          passed   (int)    — checks que passaram
          failed   (int)    — checks que falharam
          results  (object) — resultado bruto do GE (para save_quality_report)

    Raises:
        RuntimeError: se fail_pipeline_on_error=true e algum check falhar.
        ImportError:  se great-expectations não estiver instalado.
    """
    gx, gxe = _import_ge()
    logger = get_logger('quality_checks_module', logging_config=logging_config)

    # ler parâmetros de qualidade do config
    suite_name = config.get('suite_name', 'default_suite')
    fail_on_error = config.get('fail_pipeline_on_error', True)
    table_expectations = config.get('table_expectations', [])
    column_expectations = config.get('column_expectations', [])

    total_exp = len(table_expectations) + sum(len(v) for v in column_expectations.values())
    logger.info('Iniciando verificações de qualidade: %d checks definidos', total_exp)

    # construir contexto efêmero e suite
    context, batch_def, suite = _build_ephemeral_context(df, suite_name)

    # popular suite com expectations do config
    _populate_suite_with_expectations(
        suite,
        table_expectations,
        column_expectations,
        gxe
    )

    # cria e registra a ValidationDefinition
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"{suite_name}_validation",
            data=batch_def,
            suite=suite
        )
    )

    # executar a validação
    logger.info('Executando validação de qualidade...')
    results = validation_def.run(
        batch_parameters={
            'dataframe': df
        }
    )

    # sumarização dos resultados
    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info('-'*60)
    logger.info('Resultados da validação de qualidade:')
    logger.info('  Total checks: %d', total)
    logger.info('  Checks passados: %d', passed)
    logger.info('  Checks falhados: %d', failed)
    logger.info('-'*60)

    # log detalhado por expectation
    for r in results.results:
        status   = "OK  " if r.success else "FAIL"
        exp_type = r.expectation_config.type
        col      = r.expectation_config.kwargs.get("column", "(tabela)")
        logger.info("  [%s] %-50s  col=%-25s", status, exp_type, col)

    # Falha explícita se configurado — comportamento de produção
    if fail_on_error and not success:
        raise RuntimeError(
            f"Qualidade de dados REPROVADA: {failed}/{total} checks falharam.\n"
            "Revise o quality.yaml ou investigue os dados de entrada.\n"
            "Para continuar mesmo com falhas, ajuste fail_pipeline_on_error: false"
        )

    return {
        "success": success,
        "total":   total,
        "passed":  passed,
        "failed":  failed,
        "results": results,
    }

def save_quality_report(
    summary: dict[str, Any],
    output_dir: Path,
    logging_config: dict[str, Any] | None = None,
) -> Path:
    """
    Serializa o resumo de qualidade em JSON legível.

    Apenas o summary é serializado — o objeto results do GE não é
    diretamente JSON-serializable. Informações de cada check são extraídas
    e normalizadas aqui.

    Args:
        summary:        Retorno de run_quality_checks().
        output_dir:     Diretório onde salvar o relatório.
        logging_config: Seção 'logging' do pipeline.yaml.

    Returns:
        Path do arquivo JSON gerado.
    """
    logger = get_logger("quality", logging_config or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"quality_report_{timestamp}.json"

    details = []
    for r in summary["results"].results:
        # Extrai kwargs sem a chave 'column' para não duplicar
        raw_kwargs = dict(r.expectation_config.kwargs)
        column     = raw_kwargs.pop("column", None)

        # result pode conter estatísticas observadas (ex.: % nulos, contagem)
        raw_result = r.result if isinstance(r.result, dict) else {}

        details.append({
            "type":    r.expectation_config.type,
            "column":  column,
            "success": r.success,
            "kwargs":  raw_kwargs,
            "result":  {k: v for k, v in raw_result.items()
                        if not isinstance(v, (list, dict)) or len(str(v)) < 500},
        })

    report = {
        "success": summary["success"],
        "total":   summary["total"],
        "passed":  summary["passed"],
        "failed":  summary["failed"],
        "details": details,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Relatório de qualidade salvo: %s", report_path)
    return report_path