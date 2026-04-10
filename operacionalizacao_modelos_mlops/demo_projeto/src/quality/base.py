"""
src/quality/base.py — Contratos abstratos do módulo de qualidade.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O que é uma ABC (Abstract Base Class)?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uma ABC é uma classe que define um CONTRATO — uma lista de métodos que toda
subclasse OBRIGATORIAMENTE deve implementar. Ela não pode ser instanciada
diretamente; serve apenas como "molde".

Analogia: pense em uma tomada elétrica. A tomada é a ABC: define o formato
dos pinos (o contrato). Qualquer aparelho (subclasse) que queira se conectar
precisa respeitar esse formato. Não importa se é um carregador de celular ou
uma televisão — ambos seguem o mesmo contrato.

Em Python, ABCs vêm do módulo `abc`:
  - `ABC`            → herdar desta classe torna a sua classe abstrata
  - `@abstractmethod`→ decorador que marca um método como obrigatório

Exemplo mínimo:
    from abc import ABC, abstractmethod

    class Animal(ABC):           # contrato
        @abstractmethod
        def falar(self) -> str:  # método obrigatório
            ...

    class Cachorro(Animal):      # implementação concreta
        def falar(self) -> str:
            return "Au!"

    Animal()    # ← TypeError! Não pode instanciar a ABC diretamente.
    Cachorro()  # ← OK. Cachorro cumpre o contrato.

Por que usamos ABCs aqui?
─────────────────────────
Queremos que o módulo de qualidade seja EXTENSÍVEL sem ser MODIFICADO
(Princípio Aberto/Fechado do SOLID). Se amanhã precisarmos de um validador
que use Pandera, Deequ ou qualquer outra biblioteca, basta criar uma nova
subclasse de `QualityValidator` — sem mudar uma linha do código existente.

O PipelineContext só conhece `QualityValidator` (a ABC), não sabe nada sobre
Great Expectations. Isso é Inversão de Dependência (o "D" do SOLID).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ExpectationResolver(ABC):
    """
    Contrato para resolução dinâmica de classes de expectation.

    Traduz um nome (string vindo do YAML) para a classe concreta
    correspondente na biblioteca de validação escolhida.

    Subclasses concretas:
        GeExpectationResolver — resolve para classes do Great Expectations.
    """

    @abstractmethod
    def resolve(self, type_name: str) -> type:
        """
        Resolve o nome de uma expectation para sua classe concreta.

        Args:
            type_name: Nome em snake_case ou PascalCase (ex.: "expect_column_values_to_not_be_null").

        Returns:
            A classe da expectation pronta para ser instanciada.

        Raises:
            AttributeError: se o nome não corresponder a nenhuma expectation conhecida.
        """


class QualityValidator(ABC):
    """
    Contrato para validação de qualidade de dados.

    Subclasses concretas recebem um logger no construtor (injeção de dependência)
    e implementam `validate()` com a lógica específica da biblioteca escolhida.

    Subclasses concretas:
        GreatExpectationsValidator — usa GE para executar as validações.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        """
        Valida o DataFrame com base nas regras definidas em config.

        Args:
            df:     DataFrame a ser validado.
            config: Dicionário unificado com as seções do quality.yaml.

        Returns:
            Dicionário com as chaves:
              success  (bool)   — True se todos os checks passaram
              total    (int)    — número total de checks executados
              passed   (int)    — checks que passaram
              failed   (int)    — checks que falharam
              results  (Any)    — objeto de resultados brutos da biblioteca

        Raises:
            RuntimeError: se fail_pipeline_on_error=true e algum check falhar.
        """


class QualityReportWriterBase(ABC):
    """
    Contrato para persistência do relatório de qualidade.

    Separar a escrita do relatório da lógica de validação segue o
    Princípio de Responsabilidade Única (SRP): cada classe faz apenas uma coisa.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        """
        Serializa o resumo de qualidade em um arquivo.

        Args:
            summary:    Retorno de QualityValidator.validate().
            output_dir: Diretório de destino do relatório.

        Returns:
            Path absoluto do arquivo gerado.
        """
