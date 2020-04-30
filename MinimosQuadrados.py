## BIBLIOTECAS

#pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.linear_model import LinearRegression
#pip install pandas
import pandas as pd
#pip install numpy
import numpy as np
#pip install matplotlib
import matplotlib.pyplot as plt

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

#troca para categorico para fazer a classificação
def toCategorical(tipo):
    if tipo == 0:
        return 'setosa'
    elif tipo == 1:
        return 'versicolor'
    else:
        return 'virginica'

objetivo['target'] = objetivo['target'].apply(toCategorical)

# acrescenta a nova coluna ao dataset
dados = pd.concat([dados, objetivo], axis=1)

dados.drop('target', axis=1, inplace=True)

objetivo = pd.DataFrame(iris.target, columns=['target'])

dados = pd.concat([dados, objetivo], axis=1)

#seleciona apenas os dados das features / variáveis independentes
X = dados.drop(labels='sepal length (cm)', axis= 1)

#seleciona apenas os dados target / variável dependente
Y = dados['sepal length (cm)']

## CONSTRUÇÃO DO MODELO

#separação do conjunto de treino e teste usando 80 treino/20 teste e usando estado randômico a cada execução
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.2, random_state=None)

#criação do modelo de Regressão Linear
modelo = LinearRegression()

#treinamento do modelo
modelo.fit(X_treino, Y_treino)

#predicoes (objetivo)
predicoes = modelo.predict(X_teste)

## AVALIAÇÃO DO MODELO CRIADO

erro_medio_absoluto = mean_absolute_error(Y_teste, predicoes)
erro_medio_quadrado = mean_squared_error(Y_teste, predicoes)
raiz_erro_medio_quadrado = np.sqrt(mean_squared_error(Y_teste, predicoes))
