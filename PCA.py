## BIBLIOTECAS

#pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.decomposition import PCA
#pip install pandas
import pandas as pd
#pip install numpy
import numpy as np

## MINERANDO OS DADOS E SEPARANDO AS VARIÁVEIS DEPENDENTES E INDEPENDENTES

#carregando o dataset iris
iris = load_iris()

#informações sobre o dataset
#print(load_iris.__doc__)

#colunas / features
dados = pd.DataFrame(iris.data, columns=iris.feature_names)

#objetivo / target
objetivo = pd.DataFrame(iris.target, columns=['target'])

#nomes dos targets
objetivo_nomes = pd.DataFrame(iris.target_names, columns=['target_names'])

#seleciona apenas os dados das features / variáveis independentes
X = dados

#seleciona apenas os dados target / variável dependente
Y = objetivo

## CONSTRUÇÃO DO MODELO

#separação do conjunto de treino e teste usando 80 treino/20 teste e usando estado randômico a cada execução
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.2, random_state=None)

#criação do modelo de PCA (Com 3 classificações possiveis: setosa, versicolor, virginica)
modelo = PCA(n_components=3)

#treinamento do modelo
modelo_treinado = modelo.fit(X_treino)

#nova matriz
X = modelo_treinado.transform(X)

## AVALIAÇÃO DO MODELO CRIADO

#indica qual dos targets são responsáveis pela variação dos dados percentualmente
variancia_explicada = modelo.explained_variance_ratio_ 
