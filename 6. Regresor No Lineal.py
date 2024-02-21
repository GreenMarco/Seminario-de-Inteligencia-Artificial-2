# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 07:17:05 2024
@author: Marco Barrones
SIA2 - Carlos Villaseñor


arboles dedecision
knn
%matplotlib auto
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
"""
#%%
#Importaciones
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


#%%
#Variables - - - - - - - - - - - - - - - - - - - - -
np.random.seed(42)
m = 300 #muestras
r = 0.5 #ruido
ruido = r * np.random.randn(m,1)
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x ** 2 + x + 2 + ruido

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

#%%
#Modelo Arboles de Decision 1- - - - - - - - - - - - - 
model = DecisionTreeRegressor(max_depth=5)#un modelo 

#%%
#Modelo Arboles de Decision 2- - - - - - - - - - - - - 
model = DecisionTreeRegressor(min_samples_leaf=1)#otro modelo

#%%
#Modelo KNN  - - - - - - - - - - - - - - - - - - - - -
model = KNeighborsRegressor(n_neighbors=200)

#%%
#Modelo Máquina de Soporte Vectorial SVR - - - - - - - -
model = SVR(gamma='scale', C=10, kernel='rbf')#C se cambia solo en escalas de 10: 0.1, 1, 10, 100, etc.
#si cambias la gamma a numeros altos se sobreentrena y pequeño se subentrena, en ves de 'scale' poner 10 por ejemplo.

#%%
#MLP - - - - - - - - - - - - - - - - - - - - - - - - - -
model = MLPRegressor(hidden_layer_sizes=100,
                     # solver='adam',
                     # activation='relu',
                     #batch_size=10,
                     max_iter=2000, learning_rate_init=0.1)


#%%
#Train Test
model.fit(xtrain, ytrain.ravel())

print('Train', model.score(xtrain,ytrain.ravel()))
print('Test', model.score(xtest,ytest.ravel()))

xnew = np.linspace(-3, 3, 500).reshape(-1, 1)
ynew = model.predict(xnew)

#%%
#Dibujo  - - - - - - - - - - - - - - - - - - - - - - - - -
plt.plot(xnew, ynew, 'k-', linewidth=3)
plt.plot(xtrain, ytrain, 'b.')
plt.plot(xtest, ytest, 'r.')
plt.xlabel(r'$x$', fontsize = 18)
plt.xlabel(r'$y$', fontsize = 18)
plt.title('Regresor No Lineal')
plt.axis([-3,3,0,10])
plt.show()
