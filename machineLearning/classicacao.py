# Base de dados do tcc do Eduardo Franciscon
# Analise exploratória e tratamento dos dados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

dataset = pd.read_excel('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/BD Completo.xlsx')

dataset.drop(labels = ['EV/EBITDA', 'DPA', 'Dividend Yield', 'Payout', 'Luc. Liq * NR', 
                      'Resultado Bruto', 'Margem Bruta', 'EBIT', 'D&A', 'EBITDA', 
                      'Margem EBITDA', 'Res. Financeiro', 'ROA', 'SSS', 'RIF', 
                      'Margem Bancaria', 'Indc. Eficiencia', 'Indc. Basileia', 'PDD',
                      'PDD/LL', 'Equity Multi.', 'Div Liquida/EBITDA', 'Indice de Cobertura',
                      'Patri. Liquido', 'Despesas com juros', 'Custo % da divida', 'IPL', 'FCO', 'FCI',
                      'FCF', 'FCT', 'FCL', 'FCI/LL', 'CAPEX', 'FCL CAPEX', 'CAPEX/LL', 'CAPEX/FCO', 
                      'Majoritar.'],
                      axis = 1, inplace = True)



dataset.fillna(dataset.mean(), inplace=True)
dataset.dropna(inplace=True)


def corrige_segmento(texto):
  segmento = ''
  if texto == 'acessórios':
    segmento = 'acessorios'
  elif texto == 'agriculltura':
    segmento = 'agricultura'
  elif texto == 'alimentos diversos':
    segmento = 'alimentos'
  elif texto == 'eletrodomésticos':
    segmento = 'eletrodomesticos'
  elif texto == 'equipamentos e servicos':
    segmento = 'equipamentos'
  elif texto == 'mateial rodoviario':
    segmento = 'material rodoviario'
  elif texto == 'ser med hospit analises e diagnosticos' or texto == 'serv med hospit analises e disgnosticos' or texto == 'serv.med.hospit.analises e diagnosticos':
    segmento = 'hospitalar'
  elif texto == 'serviços de apoio e armazenamento':
    segmento = 'serviços de apoio e armazenagem'
  elif texto == 'serviços diversos s.a ctax':
    segmento = 'serviços diversos'
  elif texto == 'siderurgia':
    segmento = 'siderurgica'
  elif texto == 'soc. Credito e financiamento' or texto == 'soc credito e financiamento':
    segmento = 'credito'
  elif texto == 'tansporte aereo':
    segmento = 'transporte aereo'
  else:
    segmento = texto  

  return segmento

dataset['Segmento'] = dataset['Segmento'].apply(corrige_segmento)
# print(np.unique(dataset['Segmento'], return_counts=True))



def corrige_categoria(texto):
  categoria = ''
  if texto == 'crescimento ':
    categoria = 'crescimento'
  else:
    categoria = texto
  
  return categoria


dataset['Categoria'] = dataset['Categoria'].apply(corrige_categoria)

# print(np.unique(dataset['Categoria'], return_counts=True))


####### Correlação entre atributos


dataset.drop(['Rec. Liquida', 'Caixa'], axis=1, inplace=True)
dataset.drop(['Divida bruta', 'LPA', 'Caixa.1'], axis=1, inplace=True)
dataset.drop(['At. Circulante', 'Liq. Corrente'], axis = 1, inplace = True)



###### Variáveis dummys

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

y = dataset['Situação'].values
empresa = dataset['Empresa']
x_cat = dataset[['Segmento', 'Categoria']]


x_cat = onehotencoder.fit_transform(x_cat).toarray()
x_cat = pd.DataFrame(x_cat)

dataset_original = dataset.copy();

dataset.drop(['Segmento', 'Categoria', 'Situação', 'Empresa'], axis=1, inplace=True)


dataset.index = x_cat.index

dataset = pd.concat([dataset, x_cat], axis=1)

### Normalização

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_normalizado = scaler.fit_transform(dataset)
x = dataset_normalizado.copy();



########### Árvore de decisão e redes  neurais - validação cruzada
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# resultados_forest = []
# resultados_neural = []

# for i in range(30):
#   kfold = KFold(n_splits=10, shuffle=True, random_state=i)

#   random_forest = RandomForestClassifier();
#   scores = cross_val_score(random_forest, x, y, cv=kfold)
#   resultados_forest.append(scores.mean())


# for i in range(30):
#   kfold = KFold(n_splits=10, shuffle=True, random_state=i)

#   network = MLPClassifier(hidden_layer_sizes=(175, 175))
#   scores = cross_val_score(network, x, y, cv=kfold)
#   resultados_neural.append(scores.mean())


# resultados_forest = np.array(resultados_forest)
# resultados_neural = np.array(resultados_neural)


#######################################################################################################
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.2, random_state=1)


random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
random_forest.fit(x_treinamento, y_treinamento)
previsoes_random_forest = random_forest.predict(x_teste)
accuracy_score_random_forest = accuracy_score(y_teste, previsoes_random_forest) #61.42%


network_reural = MLPClassifier(hidden_layer_sizes=(175, 175))
network_reural.fit(x_treinamento, y_treinamento)
previsoes_neural = network_reural.predict(x_teste)
accuracy_score_network_reural = accuracy_score(y_teste, previsoes_neural) # 58.57
print(accuracy_score_network_reural)