# forecasting_workflow_engine/modeling/hyperparam_optimization.py
import numpy as np
import optuna
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# -------------------------------
# Otimização Prophet
# -------------------------------
def optimize_prophet_params(df, n_trials: int = 20, n_splits: int = 3):
    """
    Otimiza hiperparâmetros do Prophet usando TimeSeriesSplit.
    
    Args:
        df: DataFrame com colunas 'ds' e 'y'.
        n_trials: número de tentativas da otimização.
        n_splits: número de splits para validação cruzada temporal.

    Returns:
        dict: melhores parâmetros encontrados.
    """
    def objective(trial):
        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        interval_width = trial.suggest_float("interval_width", 0.7, 0.95)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mses = []

        for train_idx, val_idx in tscv.split(df):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            model = Prophet(
                changepoint_prior_scale=cps,
                interval_width=interval_width,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True
            )
            model.fit(train_df)
            y_pred = model.predict(val_df[['ds']])['yhat']
            mses.append(mean_squared_error(val_df['y'], y_pred))

        return np.mean(mses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# -------------------------------
# Otimização ARIMA
# -------------------------------
def optimize_arima_order(y, n_trials: int = 20, p_range=(0,5), d_range=(0,2), q_range=(0,5)):
    """
    Otimiza ordem do ARIMA (p,d,q) usando Optuna.

    Args:
        y: série temporal univariada.
        n_trials: número de tentativas da otimização.
        p_range, d_range, q_range: intervalos para p,d,q.

    Returns:
        tuple: melhor ordem (p,d,q).
    """
    def objective(trial):
        p = trial.suggest_int("p", *p_range)
        d = trial.suggest_int("d", *d_range)
        q = trial.suggest_int("q", *q_range)
        try:
            model = ARIMA(y, order=(p,d,q))
            model_fit = model.fit()
            y_pred = model_fit.fittedvalues
            mse = mean_squared_error(y, y_pred)
        except:
            mse = np.inf
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    return best["p"], best["d"], best["q"]
