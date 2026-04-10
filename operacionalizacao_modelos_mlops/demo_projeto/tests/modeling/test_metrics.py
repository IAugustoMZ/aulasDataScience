"""
test_metrics.py — Testes unitários para src/modeling/metrics.py.

Cobre:
  • calcular_metricas   — valores conhecidos, borda (predição perfeita, constante)
  • agregar_metricas_folds — média e desvio padrão de métricas por fold
"""
import numpy as np
import pytest

from src.modeling.metrics import calcular_metricas, agregar_metricas_folds


class TestCalcularMetricas:
    """Testes para a função calcular_metricas."""

    def test_predicao_perfeita(self) -> None:
        """Predição igual ao alvo deve retornar rmse=0, mae=0, r2=1, mape=0."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mets = calcular_metricas(y, y)

        assert mets['rmse'] == pytest.approx(0.0, abs=1e-9)
        assert mets['mae']  == pytest.approx(0.0, abs=1e-9)
        assert mets['r2']   == pytest.approx(1.0, abs=1e-9)
        assert mets['mape'] == pytest.approx(0.0, abs=1e-9)

    def test_valores_conhecidos(self) -> None:
        """Verifica métricas para erros simples conhecidos."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])   # erros: +10, -10, 0

        mets = calcular_metricas(y_true, y_pred)

        rmse_esperado = np.sqrt((100 + 100 + 0) / 3)
        mae_esperado  = (10 + 10 + 0) / 3
        mape_esperado = (10/100 + 10/200 + 0/300) / 3 * 100

        assert mets['rmse'] == pytest.approx(rmse_esperado, rel=1e-6)
        assert mets['mae']  == pytest.approx(mae_esperado,  rel=1e-6)
        assert mets['mape'] == pytest.approx(mape_esperado, rel=1e-6)

    def test_predicao_constante(self) -> None:
        """Predição constante deve retornar r2 negativo (ou zero)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, 2.5)  # média

        mets = calcular_metricas(y_true, y_pred)
        assert mets['r2'] == pytest.approx(0.0, abs=1e-9)
        assert mets['rmse'] > 0

    def test_chaves_retornadas(self) -> None:
        """Garante que todas as chaves esperadas estão presentes."""
        mets = calcular_metricas(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert set(mets.keys()) == {'rmse', 'mae', 'r2', 'mape'}


class TestAgregarMetricasFolds:
    """Testes para a função agregar_metricas_folds."""

    def test_media_e_std_simples(self) -> None:
        """Verifica média e desvio padrão de RMSEs conhecidos."""
        import numpy as np
        fold_mets = [
            {'rmse': 10.0, 'mae': 5.0, 'r2': 0.9, 'mape': 5.0},
            {'rmse': 20.0, 'mae': 10.0, 'r2': 0.8, 'mape': 10.0},
        ]
        agg = agregar_metricas_folds(fold_mets)

        # std calculado pela própria implementação (pode ser ddof=0 ou ddof=1)
        std_esperado = np.std([10.0, 20.0])   # ddof padrão (0)
        std_alternativo = np.std([10.0, 20.0], ddof=1)

        assert agg['cv_rmse_mean'] == pytest.approx(15.0, rel=1e-6)
        assert agg['cv_rmse_std'] in (
            pytest.approx(std_esperado,    rel=1e-4),
            pytest.approx(std_alternativo, rel=1e-4),
        ) or agg['cv_rmse_std'] == pytest.approx(std_esperado, rel=1e-4) or \
               agg['cv_rmse_std'] == pytest.approx(std_alternativo, rel=1e-4)
        assert agg['cv_r2_mean']   == pytest.approx(0.85, rel=1e-6)

    def test_fold_unico(self) -> None:
        """Com um único fold, média deve ser igual ao valor do fold."""
        fold_mets = [{'rmse': 42.0, 'mae': 21.0, 'r2': 0.95, 'mape': 7.0}]
        agg = agregar_metricas_folds(fold_mets)

        assert agg['cv_rmse_mean'] == pytest.approx(42.0, rel=1e-6)
        # std com fold único pode ser nan (ddof=1) ou 0 (ddof=0) — ambos aceitáveis
        import math
        assert math.isnan(agg['cv_rmse_std']) or agg['cv_rmse_std'] == pytest.approx(0.0, abs=1e-9)

    def test_chaves_retornadas(self) -> None:
        """Garante que as chaves cv_{metric}_{mean|std} existem."""
        fold_mets = [
            {'rmse': 1.0, 'mae': 0.5, 'r2': 0.9, 'mape': 2.0},
            {'rmse': 2.0, 'mae': 1.0, 'r2': 0.8, 'mape': 4.0},
        ]
        agg = agregar_metricas_folds(fold_mets)
        for metrica in ('rmse', 'mae', 'r2', 'mape'):
            assert f'cv_{metrica}_mean' in agg
            assert f'cv_{metrica}_std'  in agg
