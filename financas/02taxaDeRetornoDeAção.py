import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf

# Formula Taxa de retorno simples
# RS = Preço final - Preço inicial / Preço inicial * 100
# Considerando UMA ação e SEM dividendos
# Com dividendos RS =  Preço final + Dividendo - Preço inicial / Preço inicial * 100


dataset = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv')


# Taxa de Retorno de todas as ações

for i in dataset.columns[1:]:
  rs = (dataset[i][len(dataset) -1] - dataset[i][0] ) / dataset[i][0] * 100
  


# taxa de retorno diário

dataset['RS GOL'] = (dataset['GOL'] / dataset['GOL'].shift(1)) -1 
dataset['RS CVC'] = (dataset['CVC'] / dataset['CVC'].shift(1)) - 1
dataset['RS WEGE'] = (dataset['WEGE'] / dataset['WEGE'].shift(1)) - 1
dataset['RS MGLU'] = (dataset['MGLU'] / dataset['MGLU'].shift(1)) - 1
dataset['RS TOTS'] = (dataset['TOTS'] / dataset['TOTS'].shift(1)) - 1
dataset['RS BOVA'] = (dataset['BOVA'] / dataset['BOVA'].shift(1)) - 1


# Taxa de Retorno anual

# print((dataset['RS GOL'].mean() * 346) * 100) 
# print((dataset['RS CVC'].mean() * 246) * 100)
# print((dataset['RS WEGE'].mean() * 246) * 100)
# print((dataset['RS MGLU'].mean() * 246) * 100)
# print((dataset['RS TOTS'].mean() * 246) * 100)
# print((dataset['RS BOVA'].mean() * 246) * 100)



# TAXA DE RETORNO LOGARÍTMICA

# UMA AÇÃO SEM DIVIDENDOS
# RL = LOG PREÇO FINAL/PREÇO INICIAL * 100

# COM DIVIDENDOS
# RL = PREÇO FINAL + DIVIDENDO / PREÇO INICIAL * 100


# print(np.log(dataset['GOL'][len(dataset) -1] / dataset['GOL'][0]) * 100)
# print(np.log(dataset['CVC'][len(dataset) - 1] / dataset['CVC'][0]) * 100)
# print(np.log(dataset['WEGE'][len(dataset) - 1] / dataset['WEGE'][0]) * 100)
# print(np.log(dataset['MGLU'][len(dataset) - 1] / dataset['MGLU'][0]) * 100)
# print(np.log(dataset['TOTS'][len(dataset) - 1] / dataset['TOTS'][0]) * 100)
# print(np.log(dataset['BOVA'][len(dataset) - 1] / dataset['BOVA'][0]) * 100)



# Retorno logaritimo diário e anual

# Diário
dataset['RL GOL'] = np.log(dataset['GOL'] / dataset['GOL'].shift(1))
dataset['RL CVC'] = np.log(dataset['CVC'] / dataset['CVC'].shift(1))
dataset['RL WEGE'] = np.log(dataset['WEGE'] / dataset['WEGE'].shift(1))
dataset['RL MGLU'] = np.log(dataset['MGLU'] / dataset['MGLU'].shift(1))
dataset['RL TOTS'] = np.log(dataset['TOTS'] / dataset['TOTS'].shift(1))
dataset['RL BOVA'] = np.log(dataset['BOVA'] / dataset['BOVA'].shift(1))


# ANUAL
# (dataset['RL CVC'].mean() * 246) * 100
# (dataset['RL WEGE'].mean() * 246) * 100
# (dataset['RL MGLU'].mean() * 246) * 100
# (dataset['RL TOTS'].mean() * 246) * 100
# (dataset['RL BOVA'].mean() * 246) * 100


# Retorno de carteira de ações

dataset2 = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv')


dataset_normalizado = dataset2.copy()

for i in dataset_normalizado.columns[1:]:
  dataset_normalizado[i] = (dataset_normalizado[i] / dataset_normalizado[i][0])

dataset_normalizado.drop(labels=['Date'], axis=1, inplace=True)


retorno_carteira = (dataset_normalizado / dataset_normalizado.shift(1)) -1
#print(retorno_carteira)

#print('---------------------')

retorno_carteira_anual = (retorno_carteira.mean() * 246) * 100
#print(retorno_carteira_anual * 100)



# Definir pesos para taxas de retorno de ações

# pesos_carteira1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
# pesos_carteira2 = np.array([0.1, 0.2, 0.2, 0.4, 0.1, 0.0])

# print(retorno_carteira_anual)
# print('--------------------------------------------------------------------------')
# print(np.dot(retorno_carteira_anual, pesos_carteira1))
# print('--------------------------------------------------------------------------')
# print(np.dot(retorno_carteira_anual, pesos_carteira2))





dataset3 = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv')


dataset_normalizado2 = dataset3.copy()

for i in dataset_normalizado.columns[1:]:
  dataset_normalizado2[i] = (dataset_normalizado2[i] / dataset_normalizado2[i][0])




dataset_normalizado2['CARTEIRA'] = (dataset_normalizado2['GOL'] + dataset_normalizado2['CVC'] + dataset_normalizado2['WEGE'] + dataset_normalizado2['MGLU'] + dataset_normalizado2['TOTS']) / 5

figura = px.line(title='Comparativo carteira x BOVA')
for i in dataset_normalizado2.columns[1:]:
  figura.add_scatter(x=dataset_normalizado2['Date'], y=dataset_normalizado2[i], name=i)

figura.show()