from pmdarima import auto_arima
import os
import pandas as pd
from loguru import logger
import joblib
import matplotlib.pyplot as plt

from forecasting_workflow_engine.dataset import fetch_dataset
from forecasting_workflow_engine.get_model import get_prophet_model
from forecasting_workflow_engine.modeling.evaluator import evaluate_forecast
from forecasting_workflow_engine.modeling.hyperparam_optimization import optimize_prophet_params
from forecasting_workflow_engine.plots import plot_forecast
from forecasting_workflow_engine.experiments.mlflow_utils import (
    start_experiment,
    log_metrics,
    log_figure,
    log_dataframe,
    log_dict,
    log_model_file
)
from forecasting_workflow_engine.config import MODELS_DIR
import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*force_all_finite.*")


def train(
    dataset_name: str,
    model_type: str = "Prophet",
    optimize_params: bool = True,
    n_trials: int = 20
) -> dict:
    """
    Treina Prophet ou AutoARIMA e registra logs no MLflow.
    """
    # --- Carrega dataset ---
    X, y = fetch_dataset(dataset_name)
    if "ds" not in X.columns:
        X = pd.DataFrame({"ds": X.iloc[:, 0]})
    y_series = y.squeeze()

    # --- Inicia experimento MLflow seguro ---
    run_name = f"{dataset_name}_{model_type}"
    start_experiment("Forecasting", run_name=run_name)

    # --- Treinamento ---
    if model_type.lower() == "prophet":
        logger.info("Treinando Prophet...")

        # Otimização de hiperparâmetros
        prophet_params = None
        if optimize_params:
            prophet_params = optimize_prophet_params(pd.DataFrame({"ds": X["ds"], "y": y_series}),
                                                     n_trials=n_trials)
            logger.info(f"Hiperparâmetros ótimos: {prophet_params}")

        model = get_prophet_model(params=prophet_params)
        df_train = pd.DataFrame({"ds": X["ds"], "y": y_series})
        model.fit(df_train)
        forecast = model.predict(df_train)
        y_pred = forecast["yhat"]

        # Salva modelo
        model_file = os.path.join(
            MODELS_DIR, f"{dataset_name}_prophet_model.pkl")
        joblib.dump(model, model_file)
        log_model_file(model_file, artifact_name="prophet_model.pkl")

        # Loga parâmetros
        log_dict(prophet_params or {}, "prophet_params")

    elif model_type.lower() == "arima":
        logger.info("Treinando AutoARIMA...")

        # AutoARIMA
        model_fit = auto_arima(
            y_series.values,
            start_p=1, start_q=1, max_p=5, max_q=5,
            seasonal=False, stepwise=True, suppress_warnings=True,
            error_action='ignore'
        )
        y_pred = model_fit.predict_in_sample()

        # Salva modelo
        model_file = os.path.join(
            MODELS_DIR, f"{dataset_name}_autoarima_model.pkl")
        joblib.dump(model_fit, model_file)
        log_model_file(model_file, artifact_name="autoarima_model.pkl")

        # Loga parâmetros
        log_dict({"order": model_fit.order,
                 "seasonal_order": model_fit.seasonal_order}, "autoarima_params")

    else:
        raise ValueError(f"Modelo '{model_type}' não suportado.")

    # --- Avaliação ---
    metrics = evaluate_forecast(y_series.values, y_pred)
    log_metrics(metrics)

    # --- Log dataset e schema ---
    log_dataframe(X.head(10), "X_sample.csv")
    log_dataframe(pd.DataFrame({"y": y_series}).head(10), "y_sample.csv")
    log_dict({col: str(dtype)
             for col, dtype in X.dtypes.items()}, "dataset_schema")

    # --- Gráfico previsão vs real ---
    fig = plot_forecast(X, y_series, y_pred,
                        title=f"{dataset_name} - {model_type} Forecast")
    log_figure(fig, "forecast_plot.png")

    logger.success("Treinamento concluído")
    return metrics


# -------------------------------
# Executável via terminal
# -------------------------------
if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        dataset_name: str = "air_passengers",
        model_type: str = "Prophet",
        optimize_params: bool = True,
        n_trials: int = 20
    ):
        logger.info(
            f"Iniciando treinamento: {dataset_name} | Modelo: {model_type}")
        metrics = train(dataset_name=dataset_name, model_type=model_type,
                        optimize_params=optimize_params, n_trials=n_trials)
        logger.success(f"Métricas finais: {metrics}")

    app()
