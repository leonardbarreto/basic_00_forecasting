# plots.py
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import signal


def plot_forecast(df, y_true, y_pred, title="Forecast Plot"):
    """
    Gera um gráfico de previsão vs real e retorna a figura.

    Args:
        df: DataFrame com coluna 'ds'.
        y_true: Série real.
        y_pred: Série prevista.
        title: Título do gráfico.
    Returns:
        fig: Objeto matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["ds"], y_true, label="Real")
    ax.plot(df["ds"], y_pred, label="Predito")
    ax.set_title(title)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    return fig

# Função alternativa para plot de tendência


def plot_trend(y_series, window=12):
    """
    Plota a série original e a tendência estimada via média móvel.
    """
    rolling_mean = y_series.rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(y_series, label="Original")
    plt.plot(rolling_mean, color="orange",
             label=f"Tendência (rolling mean {window})")
    plt.title("Trend plot (alternativa)")
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()


def plot_periodogram(y_series, sampling_rate=1.0, title="Periodograma da Série Temporal"):
    """
    Plota o periodograma (densidade espectral de potência) de uma série temporal.

    Parâmetros
    ----------
    y_series : pd.Series ou np.ndarray
        Série temporal univariada.
    sampling_rate : float, opcional
        Taxa de amostragem (ex: 1 para séries diárias, 24 para horárias).
    title : str, opcional
        Título do gráfico.

    Referência teórica
    ------------------
    O periodograma é baseado na transformada discreta de Fourier (DFT) e mede
    a potência de cada componente de frequência:
        P(f) = |FFT(y)|² / N
    Ele é útil para identificar sazonalidades dominantes na série.
    """
    if hasattr(y_series, "values"):
        y_series = y_series.values

    # Remover NaNs
    y_series = y_series[~np.isnan(y_series)]

    # Calcula o periodograma
    freqs, power = signal.periodogram(y_series, fs=sampling_rate)

    # Filtra frequências nulas
    freqs, power = freqs[1:], power[1:]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power, color="steelblue")
    plt.title(title)
    plt.xlabel("Frequência [ciclos/unidade de tempo]")
    plt.ylabel("Densidade Espectral de Potência")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Identifica e loga frequência dominante
    dominant_freq = freqs[np.argmax(power)]
    period = 1 / dominant_freq if dominant_freq != 0 else np.nan
    logger.info(
        f"Frequência dominante: {dominant_freq:.4f} ({period:.2f} períodos por ciclo)")

    return freqs, power, dominant_freq, period
