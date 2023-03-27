import spacy
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import re


base = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLearningParaFinancas/stock_data.csv');

####### pré processamento dos textos
# !python -m spacy download pt  #para trabalhar com spacy em português

nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def preprocessamento(texto):
  texto = texto.lower();
  texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ' , texto)
  texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ' , texto)
  texto = re.sub(r" +", ' ', texto)

  documento = nlp(texto)
  lista = []
  for token in documento:
    lista.append(token.lemma_)

  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
  return lista



# print(preprocessamento('I Will Buy @Fulano45_34 The Apple Stock in https://www.rico.com.vc/ today. care caring'))

base['Text'] = base['Text'].apply(preprocessamento)
base['Tamanho'] = base['Text'].apply(len)


positivo = base[base['Sentiment'] == 1]
negativo = base[base['Sentiment'] == -1]

textos_positivos = positivo['Text'].tolist()
textos_positivos_string = ' '.join(textos_positivos)

textos_negativos = negativo['Text'].tolist()
textos_negativos_string = ' '.join(textos_negativos)




# Nuvem de palavras

from wordcloud import WordCloud

# plt.figure(figsize=(20,10))
# plt.imshow(WordCloud().generate(textos_positivos_string));
# plt.show()


# plt.figure(figsize=(20,10))
# plt.imshow(WordCloud().generate(textos_negativos_string));
# plt.show()




### Extração Entidades nomeadas

from spacy import displacy
from spacy.pipeline import EntityRuler

documento = nlp(textos_positivos_string)

empresas_positivas = []

for entidade in documento.ents:
  if entidade.label_ == 'ORG':
    empresas_positivas.append(entidade.text)

empresas_positivas = set(empresas_positivas)





##### extração de empresas negativas

documento2 = nlp(textos_negativos_string)

empresas_negativas = []

for entidade in documento2.ents:
  if entidade.label_ == 'ORG':
    empresas_negativas.append(entidade.text)

empresas_negativas = set(empresas_negativas)

empresas_positivas_negativas = empresas_positivas.intersection(empresas_negativas)
empresas_somente_positivas = empresas_positivas.difference(empresas_negativas)
empresas_somente_negativas = empresas_negativas.difference(empresas_positivas)

base.drop(['Tamanho'], axis=1, inplace=True)




#### Criação e treinamento do classificador
from sklearn.model_selection import train_test_split

base_treinamento, base_teste = train_test_split(base, test_size=0.3)

base_treinamento_final = []

for texto, sentimento in zip(base_treinamento['Text'], base_treinamento['Sentiment']):
  if sentimento == 1:
    dic = ({ 'POSITIVO': True, 'NEGATIVO': False })
  elif sentimento == -1:
    dic = ({ 'POSITIVO': False, 'NEGATIVO': True })
  base_treinamento_final.append([texto, dic.copy()])



####### Criação e treinamento do classificado




modelo = spacy.blank('en')


categorias = modelo.create_pipe('textcat')
categorias.add_label('POSITIVO')
categorias.add_label('NEGATIVO')
modelo.add_pipe(categorias)


historico = []
modelo.begin_training()

for epoca in range(5):
  random.shuffle(base_treinamento_final)
  erros = {}

  for batch in spacy.util.minibatch(base_treinamento_final, 512):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    modelo.update(textos, annotations, losses=erros)
    historico.append(erros)

  if epoca % 1 == 0:
    print(erros)


historico_erro = []
for i in historico:
  historico_erro.append(i.get('textcat'))

historico_erro = np.array(historico_erro)
# modelo.to_disk('modelo')