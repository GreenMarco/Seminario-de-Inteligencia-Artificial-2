"""
Created on Wed Jan 24 07:11:58 2024

@author: Marco Barrones
SIA2 - Carlos Villaseñor
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
Selección de variables

-Error absoluto medio MAE
-Error cuadratico medio MSE
-MedianAE
Coeficiente de determinacion R^2-Score
Coeficiente de varianza explicada EVC
-interpolacion
-extrapolacion
"""
#
#%%
#Importaciones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

#%%
#Leer datos
data = pd.read_csv('DataSets/home_data.csv')

#%%
#Elección de variables
y = np.asanyarray(data[['price']])
x = np.asanyarray(data.drop(columns=['id','price', 'date']))

scaler = StandardScaler()
x = scaler.fit_transform(x)

#%%
#Train test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1)

#Crear y entrenar modelo
model = Lasso(alpha=50)
model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))

#%%
#Extrar coeficientes
coef = np.abs(model.coef_.ravel())
df = pd.DataFrame()
names = np.array(data.drop(columns=['id','price', 'date']).columns)
df['names'] = names
df['coef'] = coef/np.sum(coef)
df.sort_values(by='coef', ascending=False, inplace=True)
df.set_index('names', inplace=True)
df.coef.plot(kind='bar')
# %%
