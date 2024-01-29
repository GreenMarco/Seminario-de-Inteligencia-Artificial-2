#%%
#Importaciones - - - - - - - - - - - - - -
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

#%%
#Variables - - - - - - - - - - - - - - - -
data = pd.read_csv('DataSets/countries.csv')
data_mex = data[data.country == 'Mexico']
#data_mex.plot.scatter(x='year', y='lifeExp')

#seleccion de variables
x = np.asanyarray(data_mex[['year']])
y = np.asanyarray(data_mex[['lifeExp']])

#%%
#Entrenamiento - - - - - - - - - - - - - -
model = linear_model.LinearRegression()
model.fit(x,y)

ypred = model.predict(x)
plt.scatter(x,y)
plt.plot(x, ypred, '--r')
plt.title('Regresión lineal')
plt.show()

#%%
#Resultados - - - - - - - - - - - - - - - - - -
#MAE
print('MAE: ', mean_absolute_error(y, ypred), 'años')
#MSE
print('MSE: ', mean_squared_error(y, ypred))
#MedianAE
print('MedianAE: ', median_absolute_error(y, ypred), 'años')
#R^2-Score
print('R2: ', r2_score(y, ypred), '% de prediccion')
#EVS
print('EVS: ', explained_variance_score(y, ypred), '% de prediccion')

print(model.score(x,y))








# %%
