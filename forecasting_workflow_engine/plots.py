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


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from loguru import logger

def plot_periodogram(y_series, sampling_rate=1.0, title="Periodograma da Série Temporal", verbose=True):
    """
    Plota e analisa o periodograma (densidade espectral de potência) de uma série temporal.

    Parâmetros
    ----------
    y_series : pd.Series ou np.ndarray
        Série temporal univariada.
    sampling_rate : float, opcional
        Taxa de amostragem (ex: 1 para séries diárias, 12 para mensais, 24 para horárias).
    title : str, opcional
        Título do gráfico.
    verbose : bool, opcional
        Se True, imprime a interpretação dos resultados.

    Retorna
    -------
    tuple : (freqs, power, dominant_freq, period)

    Fundamentação Teórica
    ---------------------
    O periodograma estima a *densidade espectral de potência (PSD)*, derivada da
    Transformada de Fourier Discreta (DFT):

        P(f) = |FFT(y)|² / N

    Ele permite identificar ciclos dominantes na série:
    - **Picos bem definidos** → indicam sazonalidade forte.
    - **Distribuição uniforme** → indica ruído branco ou série não periódica.
    - **Baixa frequência dominante** → tendência ou ciclo de longo prazo.
    """
    if hasattr(y_series, "values"):
        y_series = y_series.values

    y_series = np.asarray(y_series, dtype=float)
    y_series = y_series[~np.isnan(y_series)]

    # --- Calcula o periodograma ---
    freqs, power = signal.periodogram(y_series, fs=sampling_rate)
    freqs, power = freqs[1:], power[1:]  # remove frequência zero

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power, color="royalblue", lw=1.5)
    plt.title(title)
    plt.xlabel("Frequência [ciclos/unidade de tempo]")
    plt.ylabel("Densidade Espectral de Potência")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Interpretação automática ---
    dominant_idx = np.argmax(power)
    dominant_freq = freqs[dominant_idx]
    period = 1 / dominant_freq if dominant_freq > 0 else np.nan
    energy_ratio = power[dominant_idx] / np.sum(power)

    if verbose:
        logger.info(f"📈 Frequência dominante: {dominant_freq:.4f}")
        logger.info(f"📆 Período estimado: {period:.2f} unidades de tempo por ciclo")
        logger.info(f"⚡ Contribuição energética do pico: {energy_ratio*100:.2f}% da energia total")

        # --- Interpretação qualitativa ---
        if energy_ratio > 0.4:
            interpretation = (
                f"O gráfico mostra um pico dominante em frequência {dominant_freq:.3f}, "
                f"indicando uma **sazonalidade forte** com período aproximado de {period:.2f} unidades de tempo."
            )
        elif 0.1 < energy_ratio <= 0.4:
            interpretation = (
                f"O espectro apresenta um pico moderado em {dominant_freq:.3f}, "
                f"sugerindo uma **sazonalidade leve ou ciclo parcial** de {period:.2f} unidades."
            )
        else:
            interpretation = (
                "O espectro não apresenta picos marcantes, indicando uma série **dominada por ruído branco** "
                "ou sem periodicidade clara."
            )
        logger.info(f"🧭 Interpretação automática: {interpretation}")

    return freqs, power, dominant_freq, period

