import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf
import math
from scipy import stats


dataset = pd.read_csv('acoes.csv')
#print(dataset.describe())

inicio2016 = min(dataset['Date'][dataset['Date'].str.contains('2016')])
fim2016 = max(dataset['Date'][dataset['Date'].str.contains('2016')])

inicio2017 = min(dataset['Date'][dataset['Date'].str.contains('2017')])
fim2017 = max(dataset['Date'][dataset['Date'].str.contains('2017')])

inicio2018 = min(dataset['Date'][dataset['Date'].str.contains('2018')])
fim2018 = max(dataset['Date'][dataset['Date'].str.contains('2018')])

inicio2019= min(dataset['Date'][dataset['Date'].str.contains('2019')])
fim2019 = max(dataset['Date'][dataset['Date'].str.contains('2019')])

inicio2020 = min(dataset['Date'][dataset['Date'].str.contains('2020')])
fim2020 = max(dataset['Date'][dataset['Date'].str.contains('2020')])

inicio2021 = min(dataset['Date'][dataset['Date'].str.contains('2021')])
fim2021 = max(dataset['Date'][dataset['Date'].str.contains('2021')])

inicio2022 = min(dataset['Date'][dataset['Date'].str.contains('2022')])
fim2022 = max(dataset['Date'][dataset['Date'].str.contains('2022')])


print('----------2015-----------')
# print(dataset['CVC'][dataset['Date'] == '2015-01-02'], dataset['CVC'][dataset['Date'] == '2015-12-30'])
print(np.log(13.5 / 15.2) * 100)

# print(dataset['MGLU'][dataset['Date'] == '2015-01-02'], dataset['MGLU'][dataset['Date'] == '2015-12-30'])
print(np.log(0.06 / 0.23) * 100)


print('----------2016-----------')
# print(dataset['CVC'][dataset['Date'] == inicio2016], dataset['CVC'][dataset['Date'] ==  fim2016])
print(np.log(23.70 / 12.53) * 100)

# print(dataset['MGLU'][dataset['Date'] == inicio2016], dataset['MGLU'][dataset['Date'] ==  fim2016])
print(np.log(0.41 / 0.07) * 100)


print('----------2017-----------')
# print(dataset['CVC'][dataset['Date'] == inicio2017], dataset['CVC'][dataset['Date'] ==  fim2017])
print(np.log(48.50 / 23.02) * 100)

# print(dataset['MGLU'][dataset['Date'] == inicio2017], dataset['MGLU'][dataset['Date'] ==  fim2017])
print(np.log(2.20 / 0.39) * 100)


print('----------2018-----------')
# dataset['CVC'][dataset['Date'] == '2018-01-02'], dataset['CVC'][dataset['Date'] == '2018-12-28']
print(np.log(61.18 / 49.88) * 100)

# dataset['MGLU'][dataset['Date'] == '2018-01-02'], dataset['MGLU'][dataset['Date'] == '2018-12-28']
print(np.log(5.65 / 2.47) * 100)


print('----------2019-----------')
# print(dataset['CVC'][dataset['Date'] == '2019-01-02'], dataset['CVC'][dataset['Date'] == '2019-12-30'])
print(np.log(43.79 / 61.09) * 100)

# print(dataset['MGLU'][dataset['Date'] == '2019-01-02'], dataset['MGLU'][dataset['Date'] == '2019-12-30'])
print(np.log(11.92 / 5.81) * 100)


print('----------2020-----------')
#print(dataset['CVC'][dataset['Date'] == '2020-01-02'], dataset['CVC'][dataset['Date'] == '2020-11-03'])
print(np.log(12.42 / 44.70) * 100)

#print(dataset['MGLU'][dataset['Date'] == '2020-01-02'], dataset['MGLU'][dataset['Date'] == '2020-11-03'])
print(np.log(25.30 / 12.33) * 100)


print('----------2021-----------')
# print(dataset['CVC'][dataset['Date'] == inicio2021], dataset['CVC'][dataset['Date'] ==  fim2021])
print(np.log(13.42 / 20.17) * 100)

# print(dataset['MGLU'][dataset['Date'] == inicio2021], dataset['MGLU'][dataset['Date'] ==  fim2021])
print(np.log(7.22 / 25.20) * 100)


print('----------2022-----------')
# print(dataset['CVC'][dataset['Date'] == inicio2022], dataset['CVC'][dataset['Date'] ==  fim2022])
print(np.log(4.49 / 12.87) * 100)

# print(dataset['MGLU'][dataset['Date'] == inicio2022], dataset['MGLU'][dataset['Date'] ==  fim2022])
print(np.log(2.74 / 6.72) * 100)





# VARIÂNCIA
print()
print()

taxas_cvc = np.array([-11.86, 63.73, 74.52, 20.42, -33.29, -128.06, -40.74, -105.30])
media_cvc = taxas_cvc.sum() / len(taxas_cvc)

variancia_cvc = ((taxas_cvc - media_cvc) ** 2).sum() / len(taxas_cvc) ## ou para calcular a variancia pode-se utilizar taxas_cvc.var()
#print(taxas_cvc.var())
print(variancia_cvc)

taxas_mglu = np.array([-134.37, 176.76, 173.00, 82.74, 71.86, 71.87, -124.99, -89.71])
media_mglu = taxas_mglu.sum()  / len(taxas_mglu)

variancia_mglu = taxas_mglu.var()
print(variancia_mglu) #magalu apresenta um risco maior pois ela varia muito




## Desvio padrão
print()
print()

desvio_padrao_cvc  = math.sqrt(variancia_cvc)
print(desvio_padrao_cvc)

desvio_padrao_mglu = math.sqrt(variancia_mglu)
print(desvio_padrao_mglu)



# Coeficiente de variação
print()
print()

coeficiente_variação_cvc = (desvio_padrao_cvc / media_cvc) * 100
print(coeficiente_variação_cvc)

coeficiente_variacao_cvc_outro_metodo = stats.variation(taxas_cvc)* 100
# print(coeficiente_variacao_cvc_outro_metodo)


coeficiente_variação_mglu = stats.variation(taxas_mglu) * 100
print(coeficiente_variação_mglu)



#Risco médio anual

dataset.drop(labels= ['Date'], axis=1, inplace=True)

taxas_retorno = (dataset / dataset.shift(1)) -1
desvio_padrao = taxas_retorno.std() *  math.sqrt(246)
print(desvio_padrao)



# Correlação de ações

print(taxas_retorno.cov())
print(taxas_retorno.corr())

# plt.figure(figsize=(8,8))
# sns.heatmap(taxas_retorno.corr(), annot=True)
# plt.show()



# Covariância e correlação entre empresa

taxas_retorno_gol_cvc = taxas_retorno.drop(columns = ['WEGE', 'MGLU', 'TOTS', 'BOVA'])
print(taxas_retorno_gol_cvc.cov() * 246)



#################### Risco de um portifólio ######################

print('---------------Risco do portfólio-cvc-gol-----------------------')
pesos = np.array([0.5, 0.5])
print(np.dot(taxas_retorno_gol_cvc.cov() * 246, pesos))
print(np.dot(pesos, np.dot(taxas_retorno_gol_cvc.cov() * 246, pesos)))
print(math.sqrt(np.dot(pesos, np.dot(taxas_retorno_gol_cvc.cov() * 246, pesos))) * 100)

print('---------------Risco de todo o portfólio-----------------------')
pesos1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
print(np.dot(taxas_retorno.cov() * 246, pesos1))
variancia_portfolio1 = np.dot(pesos1, np.dot(taxas_retorno.cov() * 246, pesos1))
print(variancia_portfolio1)
volatilidade_portfolio1 = math.sqrt(variancia_portfolio1) * 100
print(volatilidade_portfolio1)

print('---------------Risco de todo o portfólio-BOVA-----------------------')

pesos2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
variancia_portfolio2 = np.dot(pesos2, np.dot(taxas_retorno.cov() * 246, pesos2))
print(variancia_portfolio2)
volatilidade_portfolio2 = math.sqrt(variancia_portfolio2)
print(volatilidade_portfolio2)