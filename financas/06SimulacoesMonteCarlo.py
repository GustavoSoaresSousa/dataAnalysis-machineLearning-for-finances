import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Simulações Monte Carlo para previsões de preços

# Usar dados do passado para criar simulações de vários cenários futuros

# Preço de hoje = preço de ontem * e**r
# Usamos o movimentos Browniano para modelar r


dataset = pd.read_csv('acoes.csv')
dataset = pd.DataFrame(dataset['BOVA'])

dataset_normalizado = dataset.copy()
for coluna in dataset.columns:
  dataset_normalizado[coluna] = dataset[coluna] / dataset[coluna][0]


dataset_taxa_retorno = np.log(1 + dataset_normalizado.pct_change())
dataset_taxa_retorno.fillna(0, inplace=True)


# Cálculo do Drift
# Direção que as taxas de retorno tiveram no passado

media = dataset_taxa_retorno.mean() * 100
variancia = dataset_taxa_retorno.var()
drift = media - (0.5 * variancia)


# Cálculo dos retornos diários

from scipy import stats

#Volatilidade: variável aleátoria
dias_frente = 50
simulacoes = 10

desvio_padrao = dataset_taxa_retorno.std()

Z = stats.norm.ppf(np.random.rand(dias_frente, simulacoes))
sns.histplot(Z)


# retornos diários

retornos_diarios = np.exp(drift.values + desvio_padrao.values * Z)# Taxa de retornos previstas para os próximos 50 dias



###Previsões de preços futuros 

previsoes = np.zeros_like(retornos_diarios)

previsoes[0] = dataset.iloc[-1]
for dia in range(1, dias_frente):
  previsoes[dia] = previsoes[dia - 1] * retornos_diarios[dia]


# Gráfico

grafico = px.line(title='Preços Futuros')

for i in range(len(previsoes.T)):
  grafico.add_scatter(y=previsoes.T[i], name=i)


#############################
import yfinance as yf

dataset_bova = gol_df = yf.download("BOVA11.SA", start='2020-11-04',end='2020-12-13')['Close']
print(dataset_bova)
dataset_bova.to_csv('bova_teste.csv');