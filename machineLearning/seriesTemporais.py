#### Séries temporais

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

# dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S%z')
# dataset = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv', parse_dates=['Date'], index_col='Date', date_parser = dateparse, usecols = ['Date', 'BOVA'])

# time_series = dataset['BOVA']


# ### Decomposição da série temporal
# decomposicao = seasonal_decompose(time_series, period=723)

# tendencia = decomposicao.trend
# sazonal = decomposicao.seasonal
# aleatorio = decomposicao.resid


# plt.plot(tendencia)
# plt.show()

# plt.plot(sazonal)
# plt.show()

# plt.plot(aleatorio)
# plt.show()



####Previsoes

# modelo = auto_arima(time_series, suppress_warnings=True, error_action='ignore')

# previsoes = modelo.predict(n_periods=90)

# Gráfico das previsões

# treinamento = time_series[:1002]
# teste = time_series[1002:]

# modelo2 = auto_arima(treinamento, suppress_warnings=True, error_action='ignore')

# previsoes = pd.DataFrame(modelo2.predict(n_periods=1490), index = teste.index)
# print(previsoes)



### Facebook prophet

from prophet import Prophet


dataset2 = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv', usecols=['Date', 'BOVA'])

dataset2.drop(dataset2[0])
for data in dataset2['Date']:

  dataset2['Date'] = dataset2[data[:11]]
print(dataset2)

dataset2 = dataset2[['Date', 'BOVA']].rename(columns = {'Date': 'ds', 'BOVA' : 'y'})


# modelo = Prophet()
# modelo.fit(dataset2)
# futuro = modelo.make_future_dataframe(periods=90)
# previsoes = modelo.predict(futuro)

# print(previsoes)


############################### UM ERRO INEXPLICAVÉL(Até o dia 30/01) POR CAUSA DAS DATAS 
### espero que meu eu do futuro consiga resolver esse erro 









