"""
conftest.py — Fixtures pytest compartilhadas para a suíte de testes de ingestão.

Fixtures disponíveis:
  tmp_raw_dir         : Diretório temporário vazio simulando data/raw.
  tmp_processed_dir   : Diretório temporário vazio simulando data/processed.
  sample_csv          : Pequeno arquivo CSV escrito em tmp_raw_dir.
  null_logger         : Logger que descarta todas as mensagens durante os testes.
  required_columns    : Lista das colunas obrigatórias do dataset California Housing.
"""
import logging
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    """Diretório temporário simulando data/raw."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def tmp_processed_dir(tmp_path: Path) -> Path:
    """Diretório temporário simulando data/processed."""
    d = tmp_path / "processed"
    d.mkdir()
    return d


@pytest.fixture
def null_logger() -> logging.Logger:
    """Logger que descarta todas as mensagens — mantém a saída dos testes limpa."""
    logger = logging.getLogger("test_null")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def sample_csv(tmp_raw_dir: Path) -> Path:
    """
    Escreve um pequeno CSV no estilo California Housing em tmp_raw_dir.
    Retorna o Path do arquivo criado.
    """
    df = pd.DataFrame({
        "median_house_value": [150000.0, 250000.0, 350000.0],
        "median_income":      [3.5, 5.0, 7.2],
        "housing_median_age": [20, 35, 52],
        "total_rooms":        [800, 1200, 600],
        "total_bedrooms":     [200, 300, 150],
        "population":         [500, 900, 400],
        "households":         [180, 280, 140],
        "latitude":           [37.88, 37.86, 37.85],
        "longitude":          [-122.23, -122.22, -122.24],
        "ocean_proximity":    ["NEAR BAY", "NEAR BAY", "INLAND"],
    })
    csv_path = tmp_raw_dir / "housing.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def required_columns() -> list[str]:
    """Colunas obrigatórias do schema do dataset California Housing."""
    return [
        "median_house_value", "median_income", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "latitude", "longitude", "ocean_proximity",
    ]
