# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 07:11:58 2024

@author: Marco Barrones
SIA2 - Carlos Villaseñor
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
{Titulo}

"""
#%%
#Importaciones
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#%%
#Variables
np.random.seed(42)

m=100
x = 6 *np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + 1 * np.random.rand(m,1)
#%%
#Entrenamiento
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3)


#%%
#Procesamiento


#%%
#Dibujo - - - - - - - - - - - - - - - - - -
x_new = np.linspace(-3,3,).reshape(-1,1)
y_new = 11111111111 #model.predict(x_new)

plt.plot(x_new,y_new,'r-',linewidth=2)
plt.plot(x,y,'b.')
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$y_1$', fontsize=18)
plt.axis([-3,3,0,10])
plt.show()

# %%
