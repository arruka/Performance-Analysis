import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = pd.read_csv('data.csv')
data['Дата'] = pd.to_datetime(data['Дата'])
data.set_index('Дата', inplace=True)


data = data[~data.index.duplicated(keep='first')]


data = data.asfreq('D')


data = data.fillna(data.mean())


data = data.dropna()


plt.figure(figsize=(12, 6))
plt.plot(data, label='Исходные данные')
plt.xlabel('Дата')
plt.ylabel('Количество продаж')
plt.legend(loc='best')
plt.title('Исходные данные о продажах')
plt.show()


plt.figure(figsize=(12, 6))
plot_acf(data, lags=30, ax=plt.subplot(211))
plot_pacf(data, lags=30, ax=plt.subplot(212))
plt.xlabel('Лаг')
plt.ylabel('Корреляция')
plt.suptitle('Графики автокорреляции и частичной автокорреляции')
plt.show()


model = auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)


optimal_order = model.get_params()['order']
print(f"Оптимальные параметры ARIMA: {optimal_order}")


train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]


model = ARIMA(train_data, order=optimal_order)
model_fit = model.fit()


forecast = model_fit.get_prediction(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)


forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()


plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Обучающая выборка')
plt.plot(test_data, label='Тестовая выборка')
plt.plot(test_data.index, forecast_mean, label='Прогноз')
plt.fill_between(test_data.index, forecast_conf_int['lower Количество продаж'], forecast_conf_int['upper Количество продаж'], color='gray', alpha=0.2)
plt.xlabel('Дата')
plt.ylabel('Количество продаж')
plt.legend(loc='best')
plt.title('Прогноз спроса на складе')
plt.show()


mse = mean_squared_error(test_data['Количество продаж'], forecast_mean)
mae = mean_absolute_error(test_data['Количество продаж'], forecast_mean)

prodag = data['Количество продаж'].sum()
spros = forecast_mean.sum()
kol_dney = len(test_data)
print(f"Общее количество продаж: {prodag:.2f}")
print(f"Общее количество дней прогноза: {kol_dney:.2f}")
print(f"Прогнозируемый спрос: {spros:.2f}")
print(f"Среднеквадратическая ошибка (MSE): {mse:.2f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")

residuals = test_data['Количество продаж'] - forecast_mean
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, residuals, label='Остатки')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Нулевая линия')
plt.xlabel('Дата')
plt.ylabel('Остатки')
plt.legend(loc='best')
plt.title('Анализ остатков модели ARIMA')
plt.show()
