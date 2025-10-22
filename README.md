# basic_00_forecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ai_workflow_engine and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── clustering_workflow_engine   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ai_workflow_engine a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── experiments                
    │   └── mlflow_utils.py     <- Utilities for MLflow integration: functions to start experiments, log parameters, metrics, and trained models
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py              <- Code to run model inference with trained models          
    │   ├──train.py                 <- High-level script to train models, handle datasets, and orchestrate the training pipeline
    │   ├──trainer.py               <- Functions to train models, perform cross-validation, and log parameters, metrics, and models to MLflow
    │   └── hyperparam_optimization.py <- Functions to perform hyperparameter optimization using Optuna
    │
    ├── pipeline.py             <- High-level pipeline script to run the full workflow: data preprocessing, model training, hyperparameter optimization, evaluation, and MLflow logging
    └── plots.py                <- Code to create visualizations
```

# Pipeline de Treinamento de Modelos com MLflow

Projeto básico de Rede Neural Artificial (RNA) desenvolvido em **PyTorch** para fins de aprendizado e experimentação com pipelines de *deep learning*.  
O objetivo é construir, treinar e avaliar uma rede simples utilizando dados sintéticos com suporte a *logging* e *monitoramento* estruturado, utilizando  `Optuna` e `MLflow`.


O pipeline está estruturado para ser executado via terminal, usando Typer, com todos os parâmetros em **kebab-case**.

---
# Executando o pipeline
## Comando base
```bash
python pipeline.py [OPTIONS]
```

## ⚙️ Opções principais

| Parâmetro        | Tipo | Padrão          | Descrição                                                                 |
| ---------------- | ---- | --------------- | ------------------------------------------------------------------------- |
| `--dataset-name` | str  | "air_passengers"| Nome do dataset disponível em `dataset.py` ou no diretório `data/`       |
| `--model-type`   | str  | "Prophet"       | Tipo de modelo a ser treinado: `"Prophet"` ou `"Arima"`                   |
| `--optimize-params` | bool | False         | Se `True`, executa otimização de hiperparâmetros com Optuna              |
| `--n-trials`     | int  | 20              | Número de iterações da otimização (usado quando `--optimize-params=True`) |
| `--task`         | str  | "forecasting"   | Tipo de tarefa do pipeline (atualmente `"forecasting"`)                  |

> ⚠️ **Importante:** use **kebab-case** no terminal (`--dataset-name`) e **não** `snake_case` (`--dataset_name`).

---

## 🚀 Exemplos de execução

| Dataset         | Modelo     | Comando                                                                                     |
| ---------------- | ----------- | ------------------------------------------------------------------------------------------- |
| Air Passengers   | Prophet    | `python -m forecasting_workflow_engine.modeling.train --dataset-name air_passengers --model-type Prophet` |
| Air Passengers   | AutoARIMA  | `python -m forecasting_workflow_engine.modeling.train --dataset-name air_passengers --model-type Arima` |
| Air Passengers   | Prophet (otimizado) | `python -m forecasting_workflow_engine.modeling.train --dataset-name air_passengers --model-type Prophet --optimize-params True --n-trials 20` |
| Air Passengers   | AutoARIMA (via pipeline) | `python -m forecasting_workflow_engine.pipelines.pipeline --dataset-name air_passengers --model-type Arima` |

---

## 💡 Dica
Se preferir rodar todos os experimentos em sequência e registrar automaticamente no MLflow:

```bash
python -m forecasting_workflow_engine.pipelines.pipeline --dataset-name air_passengers --model-type Prophet

> ⚠️ Importante: Use kebab-case no terminal (--dataset-name) e não snake_case (--dataset_name).

## Referências

### Técnicas de Forecasting
1. Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th Edition). Morgan Kaufmann.  
2. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd Edition). Morgan Kaufmann.  
3. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th Edition). Pearson.  
4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O’Reilly Media.

### Otimização de Hiperparâmetros (Optuna)
5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD’19).  
6. [Optuna Official Documentation](https://optuna.org/)

### MLflow
7. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, M., Konwinski, A., … & Stoica, I. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Data Engineering Bulletin.  
8. [MLflow Official Documentation](https://mlflow.org/)
