import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf


gol_df = yf.download("GOLL4.SA", start='2015-01-01')

# print(gol_df.info())
# print('---------------------------')
# print(gol_df.describe())
# print('---------------------------')
# print(gol_df[gol_df['Close'] >= 43.79]) # Maior valor
# print('---------------------------')
# print(gol_df[(gol_df['Close'] >= 1.15) & (gol_df['Close'] <= 1.16)]) # Menor valor

# gol_df.to_csv('gol.csv')

acoes = ['GOLL4.SA', 'CVCB3.SA', 'WEGE3.SA', 'MGLU3.SA', 'TOTS3.SA', 'BOVA11.SA']

acoes_df = pd.DataFrame();

# for acao in acoes: 
#   acoes_df[acao] = yf.download(acao, start='2015-01-01')['Close']


# acoes_df = acoes_df.rename(columns={ # renomendo colunas 
#   'GOLL4.SA': 'GOL', 
#   'CVCB3.SA': 'CVC', 
#   'WEGE3.SA': 'WEGE', 
#   'MGLU3.SA': 'MGLU', 
#   'TOTS3.SA': 'TOTS', 
#   'BOVA11.SA': 'BOVA'})

# acoes_df.dropna(inplace=True) # apagar registros que tem valores nulos
# acoes_df.to_csv('acoes.csv')



ver_acoes = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/acoes.csv')
# sns.histplot(ver_acoes['MGLU']) # PREÇOS FREQUENTES
# sns.boxplot(x= ver_acoes['MGLU']) # DETECTAR OUTLIERS
# ver_acoes.plot(x='Date', figsize=(15,7), title='Histórico do preço das ações') # PROGRESSO DAS AÇÕES

acoes_df_normalizado = ver_acoes.copy()

for i in acoes_df_normalizado.columns[1:]: # iterando sobre as colunas 
  acoes_df_normalizado[i] = acoes_df_normalizado[i]/acoes_df_normalizado[i][0]

# acoes_df_normalizado.plot(x='Date', figsize=(15,7), title='Histórico do preço das ações')
# plt.show()

figura = px.line(title='Histórico do preços das ações')
for i in ver_acoes.columns[1:]:
  figura.add_scatter(x=ver_acoes['Date'], y= ver_acoes[i], name=i)

figura.show()










