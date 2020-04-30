## BIBLIOTECAS

#pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
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

#criação do modelo de SVM
modelo = svm.SVC(C=100.0, kernel='rbf', degree=3, gamma='auto',
                 decision_function_shape="ovr", random_state = 0)

#treinamento do modelo
modelo_treinado = modelo.fit(X, Y)

#predicoes (objetivo)
predicoes = modelo_treinado.predict(X_teste)

## AVALIAÇÃO DO MODELO CRIADO

#matriz de confusão
matriz_confusao = confusion_matrix(Y_teste, predicoes)

#relatorio de classificacao
relat_classificacao = classification_report(Y_teste, predicoes)

#acuracia
acuracia = accuracy_score(Y_teste, predicoes)

