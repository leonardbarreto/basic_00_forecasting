# forecasting_workflow_engine/dataset.py
import pandas as pd
import typer
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from forecasting_workflow_engine.config import PROCESSED_DATA_DIR

app = typer.Typer()

# --- Criar pasta processed caso não exista ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Dataset loaders ----
def _load_air_passengers():
    file_path = PROCESSED_DATA_DIR / "air_passengers.csv"
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url)
        df.columns = ["Month", "Passengers"]
        df.to_csv(file_path, index=False)
        logger.info(f"AirPassengers dataset baixado e salvo em {file_path}")
    else:
        df = pd.read_csv(file_path)
        logger.info(f"AirPassengers dataset carregado de {file_path}")
    return df[["Month"]], df["Passengers"]

def _load_sunspots():
    file_path = PROCESSED_DATA_DIR / "sunspots.csv"
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
        df = pd.read_csv(url)
        df.columns = ["Month", "Sunspots"]
        df.to_csv(file_path, index=False)
        logger.info(f"Sunspots dataset baixado e salvo em {file_path}")
    else:
        df = pd.read_csv(file_path)
        logger.info(f"Sunspots dataset carregado de {file_path}")
    return df[["Month"]], df["Sunspots"]

def _load_covid_us():
    file_path = PROCESSED_DATA_DIR / "us_covid_daily.csv"
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv"
        df = pd.read_csv(url, parse_dates=["date"])
        df = df[["date", "cases"]].rename(columns={"date": "Date", "cases": "Cases"})
        df.to_csv(file_path, index=False)
        logger.info(f"US COVID dataset baixado e salvo em {file_path}")
    else:
        df = pd.read_csv(file_path)
        logger.info(f"US COVID dataset carregado de {file_path}")
    return df[["Date"]], df["Cases"]

DATASET_MAP = {
    "air_passengers": _load_air_passengers,
    "sunspots": _load_sunspots,
    "covid_us": _load_covid_us
}

# --- Funções públicas ---
def fetch_dataset(name: str):
    name = name.lower()
    if name not in DATASET_MAP:
        raise ValueError(f"Dataset '{name}' não suportado.")
    X, y = DATASET_MAP[name]()
    logger.info(f"Dataset '{name}' carregado com sucesso")
    return X, y

def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str):
    output_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    df_to_save = pd.concat([X, y], axis=1)
    df_to_save.to_csv(output_path, index=False)
    logger.info(f"Dataset '{dataset_name}' salvo em {output_path}")
    return output_path

# --- CLI ---
@app.command()
def main(dataset_name: str = "air_passengers"):
    logger.info(f"Iniciando processamento do dataset '{dataset_name}'")
    X, y = fetch_dataset(dataset_name)

    for i in tqdm(range(5)):
        if i == 2:
            logger.info("Etapa intermediária concluída...")

    save_dataset(X, y, dataset_name)
    logger.success(f"Dataset '{dataset_name}' processado e salvo com sucesso.")

if __name__ == "__main__":
    app()
