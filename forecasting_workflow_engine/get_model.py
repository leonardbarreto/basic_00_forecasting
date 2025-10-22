from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA



def get_prophet_model(params=None):
    """
    Retorna uma instância do Prophet (versão >=1.1, sem stan_backend)
    """
    params = params or {}
    model = Prophet(**params)
    return model

def get_arima_model(endog, order=(1, 1, 1)):
    return ARIMA(endog, order=order)
