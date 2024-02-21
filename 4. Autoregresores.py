# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 07:11:58 2024

@author: Marco Barrones
SIA2 - Carlos Villaseñor
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
Autoregresores (Analisis de Servicios de Tiempo)
"""
#%%
#Importaciones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%%
#Variables
data = pd.read_csv('DataSets/daily-min-temperatures.csv')

x = np.asanyarray(data[['Temp']])

#%%
#Entrenamiento


#%%
#Procesamiento

data2 =  pd.DataFrame(data.Temp)

p = 5
for i in range(1, p+1):
    data2 = pd.concat([data2, data.Temp.shift(-i)],axis=1)

data2 = data2[:-p]

x = np.asanyarray(data2.iloc[:,:-1])
y = np.asanyarray(data2.iloc[:,-1])

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3)
model = LinearRegression()
model.fit(xtrain,ytrain)

print('Train: ', model.score(xtrain,ytrain))
print('Test: ', model.score(xtest,ytest))

#%%
#Dibujo - - - - - - - - - - - - - - - - - -

#plt.plot(x)


#plt.show()

#%%
#Informativos
##Autocorrelacion con el p dia pasado
"""
p = 1
plt.scatter(x[p:],x[:-p])
print(np.corrcoef(x[p:].T,x[:-p].T))
"""
""" 
#Funcion de Autocorrelacion
#autocorrelation_plot(x)
"""
#plt.show()