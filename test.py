import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv('data.csv')
data['Дата'] = pd.to_datetime(data['Дата'])
data.set_index('Дата', inplace=True)


data = data[~data.index.duplicated(keep='first')]


data = data.asfreq('D')


plt.figure(figsize=(12, 6))
plt.plot(data, label='Исходные данные')
plt.xlabel('Дата')
plt.ylabel('Количество продаж')
plt.legend(loc='best')
plt.title('Исходные данные о продажах')
plt.show()

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]


model = ARIMA(train_data, order=(5, 1, 0))  # Подберите параметры (p, d, q) под вашу задачу
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
