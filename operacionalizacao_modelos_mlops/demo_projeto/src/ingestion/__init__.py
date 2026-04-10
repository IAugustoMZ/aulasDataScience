"""Pacote de ingestão — exporta as classes públicas de download e conversão."""
from src.ingestion.downloader import KaggleDownloader
from src.ingestion.parquet_writer import CsvToParquetIngester

__all__ = ["KaggleDownloader", "CsvToParquetIngester"]
