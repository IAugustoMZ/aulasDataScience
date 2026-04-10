"""
core/base.py — Classes abstratas base do pipeline.

Define os dois contratos principais:
  - PipelineStep : qualquer etapa que pode ser executada (download, ingestão, pré-processamento, …)
  - DataLoader   : qualquer componente que carrega dados de uma fonte externa

Todas as implementações concretas devem herdar de um desses ABCs para que o
PipelineContext possa despachar etapas de forma polimórfica.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import logging


class PipelineStep(ABC):
    """
    Contrato para uma etapa idempotente do pipeline.

    As subclasses recebem um logger compartilhado no momento da construção,
    portanto nunca precisam criar um próprio.

    A implementação de run() deve ser segura para chamadas repetidas
    (respeitar os flags skip_if_exists / force internamente).
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        """Executa a etapa. Levanta exceção em caso de falha irrecuperável."""


class DataLoader(ABC):
    """
    Contrato para carregamento de dados brutos de uma fonte externa para um diretório local.

    Retorna uma lista de Paths locais que foram baixados ou verificados.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def load(self, destination_dir: Path) -> list[Path]:
        """
        Busca os dados e os coloca em destination_dir.

        Args:
            destination_dir: Diretório local onde os dados devem ser salvos.

        Returns:
            Lista de Paths de todos os arquivos presentes em destination_dir após a execução.
        """
