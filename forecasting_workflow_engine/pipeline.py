from loguru import logger
from forecasting_workflow_engine.modeling.train import train

def run_pipeline(dataset_name: str, model_type: str = "Prophet",
                 optimize_params: bool = True, n_trials: int = 20):
    logger.info(f"Iniciando pipeline: {dataset_name} | Modelo: {model_type}")
    metrics = train(dataset_name=dataset_name, model_type=model_type,
                    optimize_params=optimize_params, n_trials=n_trials)
    logger.success(f"Pipeline concluído. Métricas finais: {metrics}")
    return metrics

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
        run_pipeline(dataset_name, model_type, optimize_params, n_trials)

    app()
