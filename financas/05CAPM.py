# CAPM
# DESCREVE AS RELAÇÕES ENTRE O RETORNO ESPERADO E O RISCO, COMPARANDO O PORTFÓLIO COM O MERCADO(IBOVESPA)

import pandas as pd
import numpy as np
import plotly.express as px

dataset = pd.read_csv('acoes.csv')
dataset.drop(labels='Date', axis=1, inplace=True)
dataset_normalizado = dataset.copy()
for coluna in dataset.columns:
  dataset_normalizado[coluna] = dataset[coluna] / dataset[coluna][0]

# print(dataset_normalizado)
dataset_taxa_retorno = (dataset_normalizado / dataset_normalizado.shift(1)) - 1 # taxa de retorno diária
dataset_taxa_retorno.fillna(0, inplace=True)
dataset_taxa_retorno.mean() * 246
# print(dataset_taxa_retorno)

# figura = px.scatter(dataset_taxa_retorno, x='BOVA', y='MGLU', title='BOVA x MGLU')
# figura.show()

beta, alpha = np.polyfit(x = dataset_taxa_retorno['BOVA'], y = dataset_taxa_retorno['MGLU'], deg = 1)

# print('BETA') # Curva da linha de regressão entre os retornos da bova e mlgu  / Comparação da carteira e o mercado
# print(beta)
# print('ALPHA') # Taxa de retorno anormal / excesso de retorno
# print(alpha)

# figura = px.scatter(dataset_taxa_retorno, x='BOVA', y='MGLU', title='BOVA x MGLU')
# figura.add_scatter(x=dataset_taxa_retorno['BOVA'], y=beta*dataset_taxa_retorno['BOVA'] + alpha)
# figura.show()



# BETA com covariância e variância

matriz_covariancia  = dataset_taxa_retorno.drop(columns=['GOL', 'CVC', 'WEGE', 'TOTS']).cov() * 246
cov_mglu_bova = matriz_covariancia.iloc[0,1]
variancia_bova =dataset_taxa_retorno['BOVA'].var() * 246

beta_mglu = cov_mglu_bova / variancia_bova




# CALCULO DO CAPM para uma ação
taxa_selic_historico = np.array([12.75, 14.25, 12.25, 6.5, 5.0, 2.0, 9.25, 13.75])


rm = dataset_taxa_retorno['BOVA'].mean() * 246
rf = taxa_selic_historico.mean() / 100
capm_mglu = rf +(beta_mglu * (rm - rf))

print('CAPM MAGALU')
print(capm_mglu)


# CALCULO DE BETA PARA TODA A BASE DE DADOS


betas = []
alphas = []

for ativo in dataset_taxa_retorno.columns[0:-1]:
  beta, alpha = np.polyfit(dataset_taxa_retorno['BOVA'], dataset_taxa_retorno[ativo], 1)
  betas.append(beta)
  alphas.append(alpha)

print(betas)
print(alphas)


def visualizaa_betas_alphas(betas, alphas):
  for i, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]):
    print(ativo, 'beta:', betas[i], 'alpha:', alphas[i] * 100)


# visualizaa_betas_alphas(betas, alphas)


## CALCULO DE CAPM PARA BASE DE DADOS

capm_empresas = []

for i, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]):
  capm_empresas.append(rf + (betas[i] * (rm - rf) ))


# print(capm_empresas)

def visualiza_capm(capms):
  for i, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]): 
    print(ativo, 'CAPM', capms[i] * 100) # SE ALGUEM INVESTIR NESSAS EMPRESAS VÃO TER X% DE COMPENSADO PELO RISCO DE TER INVESTIDO NESSA EMPRESA


visualiza_capm(capm_empresas)

pesos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

capm_portfolio = np.sum(capm_empresas * pesos) * 100
print(capm_portfolio) # SE VOCÊ INVESTIR NESSE PORTFÓLIO GANHARÁ X% DE RETORNO PARA COMPENSAR O RISCO DE INVESTIR NESSAS EMPRESAS


















