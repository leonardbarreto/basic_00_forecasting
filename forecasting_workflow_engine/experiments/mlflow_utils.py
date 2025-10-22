import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

# -------------------------------
# Experimento MLflow seguro
# -------------------------------
def start_experiment(experiment_name: str, run_name: str = None):
    """
    Inicia um experimento MLflow de forma segura:
    - Cria ou seleciona experimento.
    - Não inicia novo run se já houver run ativo.
    """
    # Cria ou seleciona experimento
    mlflow.set_experiment(experiment_name)

    # Checa run ativo
    active_run = mlflow.active_run()
    if active_run:
        logger.warning(f"Run ativo existente: {active_run.info.run_id}. Usando este run.")
        return active_run
    else:
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Novo run iniciado: {run.info.run_id} | Experimento: {experiment_name}")
        return run

# -------------------------------
# Logging de métricas
# -------------------------------
def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

# -------------------------------
# Logging de dicionários
# -------------------------------
def log_dict(d: dict, prefix: str = "params"):
    for k, v in d.items():
        mlflow.log_param(f"{prefix}_{k}", v)

# -------------------------------
# Logging de DataFrames (como CSV)
# -------------------------------
def log_dataframe(df: pd.DataFrame, artifact_name: str):
    temp_file = Path(artifact_name)
    df.to_csv(temp_file, index=False)
    mlflow.log_artifact(str(temp_file))
    temp_file.unlink()  # remove arquivo temporário

# -------------------------------
# Logging de figuras
# -------------------------------
def log_figure(fig: plt.Figure, artifact_name: str):
    temp_file = Path(artifact_name)
    fig.savefig(temp_file, bbox_inches="tight")
    mlflow.log_artifact(str(temp_file))
    plt.close(fig)
    temp_file.unlink()

# -------------------------------
# Logging de arquivo de modelo
# -------------------------------
def log_model_file(local_path: str, artifact_name: str):
    mlflow.log_artifact(local_path, artifact_path=artifact_name)
