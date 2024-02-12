# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:11:58 2024

@author: Marco Barrones
SIA2 - Carlos Villaseñor
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
Regresion lineal 

-Error absoluto medio MAE
-Error cuadratico medio MSE
-MedianAE
Coeficiente de determinacion R^2-Score
Coeficiente de varianza explicada EVC
-interpolacion
-extrapolacion
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

data = pd.read_csv('countries.csv')
data_mex = data[data.country == 'Mexico']


#data_mex.plot.scatter(x='year', y='lifeExp')

#seleccion de variables
x = np.asanyarray(data_mex[['year']])
y = np.asanyarray(data_mex[['lifeExp']])
#crear modelo
model = linear_model.LinearRegression()
model.fit(x,y)

ypred = model.predict(x)
plt.scatter(x,y)
plt.plot(x, ypred, '--r')
plt.title('Regresión lineal')

#MAE
from sklearn.metrics import mean_absolute_error
print('MAE: ', mean_absolute_error(y, ypred), 'años')
#MSE
from sklearn.metrics import mean_squared_error
print('MSE: ', mean_squared_error(y, ypred))
#MedianAE
from sklearn.metrics import median_absolute_error
print('MedianAE: ', median_absolute_error(y, ypred), 'años')
#R^2-Score
from sklearn.metrics import r2_score
print('R2: ', r2_score(y, ypred), '% de prediccion')
#EVS
from sklearn.metrics import explained_variance_score
print('EVS: ', explained_variance_score(y, ypred), '% de prediccion')

print(model.score(x,y))


year_to_predict = 2024  

lifeExp_prediction = model.predict([[year_to_predict]])

print(f'Predicción para el año {year_to_predict}: {lifeExp_prediction[0][0]} años')










