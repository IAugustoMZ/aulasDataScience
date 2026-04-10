"""
ingestion/downloader.py — KaggleDownloader

Implementa DataLoader para a API de Datasets do Kaggle.

Responsabilidades:
  1. Validar as credenciais do Kaggle na construção do objeto (falha rápida).
  2. Descobrir arquivos remotos quando expected_files estiver vazio.
  3. Baixar cada arquivo, tratando extração de zip quando necessário.
  4. Respeitar os flags skip_if_exists / force para idempotência.
"""
from __future__ import annotations

import fnmatch
import os
import zipfile
from pathlib import Path
import logging

from dotenv import load_dotenv

from src.core.base import DataLoader


class KaggleDownloader(DataLoader):
    """
    Realiza o download de arquivos de um dataset do Kaggle para um diretório local.

    Args:
        secrets_path    : Caminho para o arquivo .env contendo KAGGLE_USERNAME / KAGGLE_KEY.
        dataset         : Slug do Kaggle, ex: 'shibumohapatra/house-price'.
        file_pattern    : Padrão glob para descoberta de arquivos remotos, ex: '*.csv'.
        expected_files  : Lista explícita de nomes de arquivos a baixar. Quando vazia,
                          a API é consultada para descobrir os arquivos correspondentes.
        skip_if_exists  : Pula o download quando todos os arquivos esperados já existem.
        force           : Sempre re-baixa, ignorando skip_if_exists.
        logger          : Logger compartilhado do pipeline.
    """

    def __init__(
        self,
        secrets_path: Path,
        dataset: str,
        file_pattern: str = "*.csv",
        expected_files: list[str] | None = None,
        skip_if_exists: bool = True,
        force: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger or logging.getLogger(__name__))
        self.secrets_path = Path(secrets_path)
        self.dataset = dataset
        self.file_pattern = file_pattern
        self.expected_files: list[str] = expected_files or []
        self.skip_if_exists = skip_if_exists
        self.force = force

        self._validar_credenciais()

    # ── Contrato DataLoader ───────────────────────────────────────────────────

    def load(self, destination_dir: Path) -> list[Path]:
        """
        Baixa todos os arquivos esperados para destination_dir.

        Se expected_files não foi fornecido na construção, realiza a descoberta
        remota de arquivos primeiro.

        Returns:
            Lista de Paths de todos os arquivos presentes em destination_dir após a execução.
        """
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        arquivos = self._resolver_arquivos_esperados()

        self.logger.info("Dataset  : %s", self.dataset)
        self.logger.info("Padrão   : %s", self.file_pattern)
        self.logger.info("Arquivos : %s", arquivos)

        if (
            not self.force
            and self.skip_if_exists
            and self._todos_presentes(destination_dir, arquivos)
        ):
            self.logger.info(
                "%d arquivo(s) esperado(s) já presentes em '%s'. Download ignorado. "
                "(Defina force_redownload: true no pipeline.yaml para forçar o re-download.)",
                len(arquivos),
                destination_dir,
            )
            return [destination_dir / f for f in arquivos]

        api = self._autenticar()
        baixados: list[Path] = []

        for nome_arquivo in arquivos:
            dest_path = destination_dir / nome_arquivo

            if (
                not self.force
                and self.skip_if_exists
                and dest_path.exists()
                and dest_path.stat().st_size > 0
            ):
                self.logger.info(
                    "  [SKIP] '%s' já existe (%s)",
                    nome_arquivo,
                    _formatar_tamanho(dest_path.stat().st_size),
                )
                baixados.append(dest_path)
                continue

            self.logger.info("  [DOWN] Baixando '%s'...", nome_arquivo)
            try:
                api.dataset_download_file(
                    dataset=self.dataset,
                    file_name=nome_arquivo,
                    path=str(destination_dir),
                    force=True,
                    quiet=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao baixar '{nome_arquivo}' de '{self.dataset}': {exc}"
                ) from exc

            # O Kaggle pode retornar {nome_arquivo}.zip independente da extensão original
            zip_path = Path(str(dest_path) + ".zip")
            if zip_path.exists() and not dest_path.exists():
                _extrair_zip(zip_path, destination_dir, self.logger)

            if dest_path.exists():
                self.logger.info(
                    "  [OK]   '%s' salvo (%s)",
                    nome_arquivo,
                    _formatar_tamanho(dest_path.stat().st_size),
                )
                baixados.append(dest_path)
            else:
                candidatos = list(destination_dir.glob(f"*{Path(nome_arquivo).suffix}"))
                dica = f" Encontrados: {[c.name for c in candidatos]}" if candidatos else ""
                raise RuntimeError(
                    f"Download de '{nome_arquivo}' aparentemente concluído, mas arquivo não "
                    f"encontrado em '{dest_path}'.{dica}"
                )

        self.logger.info(
            "Download concluído: %d/%d arquivo(s) em '%s'",
            len(baixados),
            len(arquivos),
            destination_dir,
        )
        return baixados

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _validar_credenciais(self) -> None:
        """Carrega o .env e confirma que KAGGLE_USERNAME e KAGGLE_KEY estão definidos."""
        load_dotenv(dotenv_path=str(self.secrets_path))
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            raise EnvironmentError(
                "Credenciais do Kaggle não encontradas. "
                f"Defina KAGGLE_USERNAME e KAGGLE_KEY em '{self.secrets_path}'."
            )
        self.logger.info("Credenciais do Kaggle OK.")

    def _autenticar(self):
        """Retorna uma instância autenticada do KaggleApi."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as exc:
            raise RuntimeError(f"Falha na autenticação da API do Kaggle: {exc}") from exc

    def _resolver_arquivos_esperados(self) -> list[str]:
        """Retorna expected_files; se vazio, consulta a API do Kaggle para descobri-los."""
        if self.expected_files:
            return self.expected_files
        return self._listar_arquivos_remotos()

    def _listar_arquivos_remotos(self) -> list[str]:
        """Consulta a API do Kaggle e retorna nomes de arquivos que correspondem ao file_pattern."""
        try:
            api = self._autenticar()
            arquivos = api.dataset_list_files(self.dataset).files
            correspondentes = sorted(
                f.name for f in arquivos if fnmatch.fnmatch(f.name, self.file_pattern)
            )
            self.logger.info(
                "%d arquivo(s) remoto(s) encontrado(s) com padrão '%s' em '%s'",
                len(correspondentes),
                self.file_pattern,
                self.dataset,
            )
            return correspondentes
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao listar arquivos do dataset Kaggle '{self.dataset}': {exc}"
            ) from exc

    @staticmethod
    def _todos_presentes(diretorio: Path, nomes: list[str]) -> bool:
        """Retorna True somente se todos os arquivos existem com tamanho maior que zero."""
        return all(
            (diretorio / f).exists() and (diretorio / f).stat().st_size > 0
            for f in nomes
        )


# ── Utilitários do módulo ─────────────────────────────────────────────────────

def _formatar_tamanho(tamanho_bytes: int) -> str:
    """Formata bytes em string legível por humanos."""
    for unidade in ("B", "KB", "MB", "GB"):
        if tamanho_bytes < 1024:
            return f"{tamanho_bytes:.1f} {unidade}"
        tamanho_bytes //= 1024
    return f"{tamanho_bytes:.1f} TB"


def _extrair_zip(zip_path: Path, destination_dir: Path, logger: logging.Logger) -> None:
    """Extrai um arquivo .zip para destination_dir e remove o arquivo compactado."""
    logger.info("  [UNZIP] Extraindo '%s'...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir)
    zip_path.unlink()
    logger.info("  [UNZIP] Concluído. Arquivo compactado removido.")
