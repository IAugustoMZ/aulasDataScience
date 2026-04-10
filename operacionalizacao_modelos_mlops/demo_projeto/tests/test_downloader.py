"""
test_downloader.py — Testes unitários para KaggleDownloader.

Estratégia:
  - A API do Kaggle é sempre mockada. Nenhuma chamada de rede é realizada.
  - O carregamento do dotenv e os valores de os.environ são patcheados para que
    as credenciais possam ser injetadas ou omitidas sem tocar no sistema de arquivos.
  - Cada teste cobre um único comportamento de forma isolada.
"""
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Garante que a raiz do projeto seja importável independente de como o pytest é invocado
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.downloader import KaggleDownloader, _formatar_tamanho, _extrair_zip


# ── Helpers ───────────────────────────────────────────────────────────────────

def _criar_downloader(
    tmp_path: Path,
    null_logger,
    expected_files: list[str] | None = None,
    skip_if_exists: bool = True,
    force: bool = False,
) -> KaggleDownloader:
    """
    Cria um KaggleDownloader com credenciais mockadas para que o construtor
    não levante EnvironmentError.
    """
    secrets = tmp_path / "secrets.env"
    secrets.touch()

    with patch("src.ingestion.downloader.load_dotenv"), \
         patch.dict(os.environ, {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"}):
        return KaggleDownloader(
            secrets_path=secrets,
            dataset="owner/dataset",
            file_pattern="*.csv",
            expected_files=expected_files or ["data.csv"],
            skip_if_exists=skip_if_exists,
            force=force,
            logger=null_logger,
        )


# ── Validação de credenciais ──────────────────────────────────────────────────

class TestValidacaoCredenciais:
    def test_levanta_erro_quando_credenciais_ausentes(self, tmp_path, null_logger):
        secrets = tmp_path / "secrets.env"
        secrets.touch()

        with patch("src.ingestion.downloader.load_dotenv"), \
             patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KAGGLE_USERNAME", None)
            os.environ.pop("KAGGLE_KEY", None)

            with pytest.raises(EnvironmentError, match="Credenciais do Kaggle não encontradas"):
                KaggleDownloader(
                    secrets_path=secrets,
                    dataset="owner/dataset",
                    logger=null_logger,
                )

    def test_sucesso_quando_credenciais_presentes(self, tmp_path, null_logger):
        secrets = tmp_path / "secrets.env"
        secrets.touch()

        with patch("src.ingestion.downloader.load_dotenv"), \
             patch.dict(os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}):
            dl = KaggleDownloader(
                secrets_path=secrets,
                dataset="owner/dataset",
                expected_files=["f.csv"],
                logger=null_logger,
            )
        assert dl.dataset == "owner/dataset"


# ── Lógica de skip ────────────────────────────────────────────────────────────

class TestLogicaSkip:
    def test_pula_download_quando_todos_arquivos_presentes(self, tmp_path, null_logger, tmp_raw_dir):
        esperados = ["data.csv"]
        (tmp_raw_dir / "data.csv").write_text("col\n1\n")

        dl = _criar_downloader(tmp_path, null_logger, expected_files=esperados)
        resultado = dl.load(tmp_raw_dir)

        assert resultado == [tmp_raw_dir / "data.csv"]

    def test_force_ignora_skip(self, tmp_path, null_logger, tmp_raw_dir):
        esperados = ["data.csv"]
        (tmp_raw_dir / "data.csv").write_text("col\n1\n")

        dl = _criar_downloader(tmp_path, null_logger, expected_files=esperados, force=True)

        mock_api = MagicMock()
        mock_api.dataset_download_file.side_effect = lambda **kw: (
            (tmp_raw_dir / "data.csv").write_text("col\n2\n")
        )

        with patch.object(dl, "_autenticar", return_value=mock_api):
            dl.load(tmp_raw_dir)

        mock_api.dataset_download_file.assert_called_once()

    def test_pula_apenas_arquivo_ja_existente(self, tmp_path, null_logger, tmp_raw_dir):
        """Quando skip_if_exists=True mas apenas um de dois arquivos existe,
        somente o arquivo ausente deve ser baixado."""
        (tmp_raw_dir / "existe.csv").write_text("col\n1\n")
        esperados = ["existe.csv", "ausente.csv"]

        dl = _criar_downloader(tmp_path, null_logger, expected_files=esperados)

        mock_api = MagicMock()
        mock_api.dataset_download_file.side_effect = lambda **kw: (
            (tmp_raw_dir / kw["file_name"]).write_text("col\n1\n")
        )

        with patch.object(dl, "_autenticar", return_value=mock_api):
            dl.load(tmp_raw_dir)

        # Apenas o arquivo ausente deve ter acionado uma chamada de download
        mock_api.dataset_download_file.assert_called_once_with(
            dataset="owner/dataset",
            file_name="ausente.csv",
            path=str(tmp_raw_dir),
            force=True,
            quiet=False,
        )


# ── Descoberta remota de arquivos ─────────────────────────────────────────────

class TestDescobertaRemota:
    def test_usa_api_quando_expected_files_vazio(self, tmp_path, null_logger, tmp_raw_dir):
        secrets = tmp_path / "secrets.env"
        secrets.touch()

        with patch("src.ingestion.downloader.load_dotenv"), \
             patch.dict(os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}):
            dl = KaggleDownloader(
                secrets_path=secrets,
                dataset="owner/dataset",
                file_pattern="*.csv",
                expected_files=[],  # vazio → aciona descoberta
                logger=null_logger,
            )

        arquivo_remoto = MagicMock()
        arquivo_remoto.name = "descoberto.csv"

        mock_api = MagicMock()
        mock_api.dataset_list_files.return_value.files = [arquivo_remoto]
        mock_api.dataset_download_file.side_effect = lambda **kw: (
            (tmp_raw_dir / kw["file_name"]).write_text("col\n1\n")
        )

        with patch.object(dl, "_autenticar", return_value=mock_api):
            dl.load(tmp_raw_dir)

        mock_api.dataset_list_files.assert_called_once_with("owner/dataset")
        assert (tmp_raw_dir / "descoberto.csv").exists()


# ── Extração de zip ───────────────────────────────────────────────────────────

class TestExtracaoZip:
    def test_zip_extraido_e_removido(self, tmp_path, null_logger):
        dest = tmp_path / "dest"
        dest.mkdir()
        conteudo_csv = b"col\n1\n2\n"

        zip_path = dest / "data.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", conteudo_csv)

        _extrair_zip(zip_path, dest, null_logger)

        assert (dest / "data.csv").read_bytes() == conteudo_csv
        assert not zip_path.exists()


# ── Tratamento de erros ───────────────────────────────────────────────────────

class TestTratamentoErros:
    def test_levanta_runtime_error_em_falha_da_api(self, tmp_path, null_logger, tmp_raw_dir):
        dl = _criar_downloader(tmp_path, null_logger, expected_files=["data.csv"])

        mock_api = MagicMock()
        mock_api.dataset_download_file.side_effect = Exception("Erro de API")

        with patch.object(dl, "_autenticar", return_value=mock_api):
            with pytest.raises(RuntimeError, match="Falha ao baixar"):
                dl.load(tmp_raw_dir)

    def test_levanta_quando_arquivo_nao_encontrado_apos_download(self, tmp_path, null_logger, tmp_raw_dir):
        dl = _criar_downloader(tmp_path, null_logger, expected_files=["fantasma.csv"])

        mock_api = MagicMock()
        # download_file retorna sem criar o arquivo
        mock_api.dataset_download_file.return_value = None

        with patch.object(dl, "_autenticar", return_value=mock_api):
            with pytest.raises(RuntimeError, match="arquivo não encontrado"):
                dl.load(tmp_raw_dir)


# ── Função utilitária ─────────────────────────────────────────────────────────

class TestFormatarTamanho:
    @pytest.mark.parametrize("tamanho,esperado", [
        (500, "500.0 B"),
        (1024, "1.0 KB"),
        (1024 ** 2, "1.0 MB"),
        (1024 ** 3, "1.0 GB"),
    ])
    def test_formatar_tamanho(self, tamanho, esperado):
        assert _formatar_tamanho(tamanho) == esperado
