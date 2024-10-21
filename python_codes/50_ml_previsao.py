import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima
from prophet import Prophet


"""PREVISÃO DE VALORES DAS AÇÕES DA MAGAZINE LUIZA"""

plt.style.use('C:/Users/Edu/Desktop/Estudo/11-ML-DEFINITIVO/EXPERT_ACADEMY/2_Economia_Ml/python_codes/pitayasmoothie-dark.mplstyle')


dataset = pd.read_csv('database/acoes.csv')


dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y-%m-%d')
#adsadas


dataset.set_index('Date', inplace=True)


time_series = dataset['MGLU3.SA']

plt.figure(figsize=(12, 6))
plt.plot(time_series, linewidth=2)
plt.title('Preço de Fechamento MAGALU', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço de Fechamento', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

"""-------------------------------- PREVISOES ARIMA --------------------------------"""


n_test_days = 365
a = len(time_series) - n_test_days
treinamento = time_series[:a]
teste = time_series[a:]
n_periods = len(teste)

modelo_arima = auto_arima(treinamento, suppress_warnings=True, error_action='ignore')


previsoes_arima = modelo_arima.predict(n_periods=n_periods, return_conf_int=False, 
start=len(treinamento))

previsoes_arima = pd.DataFrame(previsoes_arima,columns=['previsoes_arima'])
previsoes_arima.index = teste.index


mae_arima = mean_absolute_error(teste, previsoes_arima['previsoes_arima'])




"""-------------------------------PREVISOES PROPHET--------------------------------"""

from prophet.plot import plot_plotly, plot_components_plotly

df_prophet = dataset.reset_index(level=0)[['Date', 'MGLU3.SA']]
df_prophet = df_prophet.rename(columns={'Date': 'ds', 'MGLU3.SA': 'y'})


modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)


futuro = modelo_prophet.make_future_dataframe(periods=n_test_days)
previsoes_prophet = modelo_prophet.predict(futuro)


previsoes_prophet_test = previsoes_prophet.set_index('ds').loc[teste.index]['yhat']


mae_prophet = mean_absolute_error(teste, previsoes_prophet_test)



"""-------------------------------VALIDAÇÃO DE MODELOS E PLOT --------------------------------"""
mae_img = mpimg.imread('C:/Users/Edu/Desktop/Estudo/11-ML-DEFINITIVO/EXPERT_ACADEMY/2_Economia_Ml/python_codes/mean_absolute_error.png')


plot_plotly(modelo_prophet,previsoes_prophet)
plot_components_plotly(modelo_prophet,previsoes_prophet)
print(f'Mean Absolute Error (ARIMA): {mae_arima}')
print(f'Mean Absolute Error (Prophet): {mae_prophet}')

# Plota os gráficos do Prophet

# Exibe os erros médios absolutos
print(f'Mean Absolute Error (ARIMA): {mae_arima}')
print(f'Mean Absolute Error (Prophet): {mae_prophet}')

# Plota o gráfico principal
plt.figure(figsize=(12, 8))
plt.plot(treinamento[1000:], label='Treino Modelo')
plt.plot(teste, label='Valores Reais', color='orange')
plt.plot(previsoes_arima, label='Previsões - Modelo não Robusto', color='grey')
plt.plot(previsoes_prophet['ds'][2000:2800], previsoes_prophet['yhat'][2000:2800], label='Previsões - Modelo Robusto', color='red')

plt.xticks(rotation=45)
plt.grid('on')
plt.legend()


ax_img = plt.axes([0.7, 0.5, 0.2, 0.2]) 
ax_img.axis('off')
ax_img.imshow(mae_img)


plt.text(150, 150, f' {mae_prophet:.2f}', fontsize=10, color='white', ha='center', va='center')
plt.show()




