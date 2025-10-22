# plots.py
import matplotlib.pyplot as plt

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
