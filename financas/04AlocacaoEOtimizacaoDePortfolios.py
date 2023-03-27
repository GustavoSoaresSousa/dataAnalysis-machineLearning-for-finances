###################################### Alocação e Otimização de portfólio #############################################################


# Definir pesos para ações
# Sharpe ratio
# Cálculo de Markowitz
# Alocação Randomica dos pesos
# Algoritimos de otimização


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

dataset = pd.read_csv('acoes.csv')

def alocacao_ativos(dataset, dinheiro_total, seed = 0, melhores_pesos = []):
  dataset2 = dataset.copy()
  if seed != 0:
    np.random.seed(seed)


  if len(melhores_pesos) > 0:
    pesos  = melhores_pesos
  else:
    pesos = np.random.random(len(dataset2.columns) -1)
    pesos = pesos / pesos.sum()

  colunas = dataset2.columns[1:]

  for coluna in colunas:
    dataset2[coluna] = (dataset2[coluna] / dataset2[coluna][0])

  for i, acao in enumerate(dataset2.columns[1:]): # enumerete ajuda a percorrer os nomes(valores) do indice e os indices de fato
    # print(i, acao) i = indices, acao = nome da ação
    dataset2[acao] = dataset2[acao] * pesos[i] * dinheiro_total

  dataset2['Total'] = dataset2.sum(axis=1)

  datas = dataset2['Date']
  dataset2.drop(labels = ['Date'], axis= 1, inplace=True)
  dataset2['Taxa de Retorno'] = 0.0
  
  for i in range(1, len(dataset2)):
    dataset2['Taxa de Retorno'][i] = (dataset2['Total'][i] / dataset2['Total'][i - 1] - 1)  * 100

  acoes_pesos = pd.DataFrame(data={'Ações': colunas, 'Pesos': pesos * 100 })
  return dataset2, datas, acoes_pesos, dataset2.loc[len(dataset2 ) -1]['Total']


# dados, datas, acoes_pesos, total = alocacao_ativos(dataset, 5000, 10)

# print('-----------Ações-------------')
# print(dados)

# print('-----------Datas-------------')
# print(datas)

# print('-----------Pesos-------------')
# print(acoes_pesos)

# print('-----------Total--------------')
# print(total)



# figura = px.line(x=datas, y=dados['Taxa de Retorno'], title='Retorno diário do portfólio')
# figura.show()

# figura = px.line(title='Evolução')
# for i in dados.drop(columns=['Total', 'Taxa de Retorno']).columns:
#   figura.add_scatter(x=datas, y=dados[i], name=i)

# figura.show()


######################## Sharpe Ratio #########################

# Calcula o retorno do investimento comparado com o risco


# retorno_acumulado = dados.loc[len(dados) -1]['Total'] - dados.loc[0]['Total'] - 1
# desvio_padrao = dados['Taxa de Retorno'].std()

taxa_selic_historico = np.array([12.75, 14.25, 12.25, 6.5, 5.0, 2.0])
# taxa_selic_historico.mean() / 100

# sharpe_ratio = (dados['Taxa de Retorno'].mean() - taxa_selic_historico.mean() / 100) / dados['Taxa de Retorno'].std() * np.sqrt(246)
# print('---------Sharpe Ratio-------------')
# print(sharpe_ratio)


# retorno = total - 5000 
# print('-----------Retorno das Ações------')
# print(retorno)

taxa_selic_2015 = 12.75
taxa_selic_2016 = 14.25
taxa_selic_2017 = 12.25
taxa_selic_2018 = 6.50
taxa_selic_2019 = 5.0
taxa_selic_2020 = 2.0
taxa_selic_2021 = 9.25
taxa_selic_2022 = 13.75

# valor_2015 = total + (total * taxa_selic_2015 / 100)
# valor_2016 = valor_2015 + (valor_2015 * taxa_selic_2016 / 100)
# valor_2017 = valor_2016 + (valor_2016 * taxa_selic_2017 / 100)
# valor_2018 = valor_2017 + (valor_2017 * taxa_selic_2018 / 100)
# valor_2019 = valor_2018 + (valor_2018 * taxa_selic_2019 / 100)
# valor_2020 = valor_2019 + (valor_2019 * taxa_selic_2020 / 100)
# valor_2021 = valor_2020 + (valor_2020 * taxa_selic_2021 / 100)
# valor_2022 = valor_2021 + (valor_2021 * taxa_selic_2022 / 100)
# print('------Retorno da renda fixa selic --------')
# print((valor_2022 - 5000) - (valor_2022 * 15 / 100))



##################Alocação de Portfólio - randômico#########################

import sys


def alocacao_portfolio(dataset, dinheiro_total, sem_risco, repeticoes):
  dataset = dataset.copy()
  dataset_original = dataset.copy()

  colunas = dataset.columns[1:]

  melhor_shape_ratio = 1 - sys.maxsize
  melhores_pesos = np.empty
  melhor_volatilidade = 0
  melhor_retorno = 0


  lista_retorno_esperado = []
  lista_volatilidade_esperada = []
  lista_sharpe_ratio = []

  for _ in range(repeticoes): #Definição dos Pesos
    pesos = np.random.random(len(dataset.columns) - 1)
    pesos = pesos / pesos.sum()

    for coluna in colunas: # Normalização
      dataset[coluna] = dataset[coluna] / dataset[coluna][0]

    for i, acao in enumerate(colunas): # Alocação do dinheiro pelos pesos
      dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total

    dataset.drop(labels = ['Date'], axis=1, inplace=True)

    retorno_carteira = np.log(dataset / dataset.shift(1)) 
    matriz_covariancia = retorno_carteira.cov() 

    dataset['Total'] = dataset.sum(axis = 1) # Somando todos os ativos em todos os dias
    dataset['Taxa Retorno'] = 0.0

    for i in range(1, len(dataset)):
      dataset['Taxa Retorno'][i] = np.log(dataset['Total'][i] / dataset['Total'][i - 1]) # Retorno Diário da Carteira
      

    #sharpe_ratio_portfolio = (dataset['Taxa Retorno'].mean() - sem_risco) / dataset['Taxa Retorno'].std() * np.sqrt(246)
    retorno_esperado = np.sum(dataset['Taxa Retorno'].mean() * pesos) * 246 # Variação Anual
    volatilidade_esperada = np.sqrt(np.dot(pesos, np.dot(matriz_covariancia * 246, pesos))) # Desvio Padrão
    sharpe_ratio = (retorno_esperado - sem_risco) / volatilidade_esperada
    if sharpe_ratio > melhor_shape_ratio:
      melhor_shape_ratio = sharpe_ratio
      melhores_pesos = pesos
      melhor_volatilidade = volatilidade_esperada
      melhor_retorno = retorno_esperado

    lista_retorno_esperado.append(retorno_esperado)
    lista_volatilidade_esperada.append(volatilidade_esperada)
    lista_sharpe_ratio.append(sharpe_ratio)

    dataset = dataset_original.copy()

  return melhor_shape_ratio, melhores_pesos, melhor_volatilidade, melhor_retorno




# sharpe_ratio_portfolio, melhores_pesos, ls_retorno, ls_volatilidade, ls_sharpe_ratio, melhor_volatilidade, melhor_retorno = alocacao_portfolio(pd.read_csv('acoes.csv'), 5000, taxa_selic_historico.mean() / 100, 1000)
# print('-------Sharpe Ratio--------')
# print(sharpe_ratio_portfolio)

# print('------Melhor volatilidade-----')
# print(melhor_volatilidade)

# print('-------Melhor Retorno--------')
# print(melhor_retorno)

#_,  _, acoes_pesos, soma_valor = alocacao_ativos(pd.read_csv('acoes.csv'), 5000, melhores_pesos=melhores_pesos)

# print('-------Ações Pesos------')
# print(acoes_pesos)
# print('------Total--------')
# print(soma_valor)






#################### Algoritimo Otimização #######################
import six
sys.modules['sklearn.externals.six'] = six
import mlrose

dataset_original = pd.read_csv('acoes.csv')
dinheiro_total = 5000
sem_risco = taxa_selic_historico.mean() / 100


def fitness_function(solucao):
  dataset = dataset_original.copy()
  pesos = solucao / solucao.sum()

  colunas = dataset.columns[1:]

  for coluna in colunas: # Normalização
    dataset[coluna] = dataset[coluna] / dataset[coluna][0]

  for i, acao in enumerate(colunas): # Alocação do dinheiro pelos pesos
    dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total

  dataset.drop(labels = ['Date'], axis=1, inplace=True)

  dataset['Total'] = dataset.sum(axis = 1) # Somando todos os ativos em todos os dias
  dataset['Taxa Retorno'] = 0.0

  for i in range(1, len(dataset)):
    dataset['Taxa Retorno'][i] = np.log(dataset['Total'][i] / dataset['Total'][i - 1]) * 100

  sharpe_ratio = (dataset['Taxa Retorno'].mean() - sem_risco) / dataset['Taxa Retorno'].std() * np.sqrt(246)

  return sharpe_ratio

np.random.seed(10)
pesos = np.random.random(len(dataset.columns) - 1)
pesos = pesos / pesos.sum()
# print(fitness_function(pesos))



def visualiza_alocação(solucao):
  colunas = dataset.columns[1:]
  for i in range(len(solucao)):
    print(colunas[i], solucao[i] * 100)

# visualiza_alocação(pesos)

## HILL CLIMB
fitness = mlrose.CustomFitness(fitness_function)
problema_maximizacao = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=True, min_val=0, max_val=1)
problema_minimizacao = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=False, min_val=0, max_val=1)

# melhor_solucao, melhor_custo = mlrose.hill_climb(problema_maximizacao, random_state=1)
# melhor_solucao = melhor_solucao / melhor_solucao.sum()

# visualiza_alocação(melhor_solucao)

# _,_,_, soma_valor = alocacao_ativos(pd.read_csv('acoes.csv'), 5000, melhores_pesos=melhor_solucao)


## pior otimização é só fazer com problema_minimizacao. ##





## Simulated annealing

melhor_solucao2, melhor_custo2 = mlrose.simulated_annealing(problema_maximizacao, random_state=1)
melhor_solucao2 = melhor_solucao2 / melhor_solucao2.sum()

# visualiza_alocação(melhor_solucao2)

# _,_,_, soma_valor = alocacao_ativos(pd.read_csv('acoes.csv'), 5000, melhores_pesos=melhor_solucao2)



## Algoritmo genético

problema_maximizacao_ag = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=True, min_val=0, max_val=1)
melhor_solucao3, melhor_custo3 = mlrose.genetic_alg(problema_maximizacao_ag, random_state=1)
melhor_solucao3 = melhor_solucao3 / melhor_solucao3.sum()


# _,_,_, soma_valor = alocacao_ativos(pd.read_csv('acoes.csv'), 5000, melhores_pesos=melhor_solucao3)
# print(soma_valor)




