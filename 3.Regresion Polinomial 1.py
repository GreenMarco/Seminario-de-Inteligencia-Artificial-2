#%%
#Importaciones
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%%
#Variables
np.random.seed(42)

m=100
x = 6 *np.random.rand(m,1) - 3
y = 0.5 * x**2 + x + 2 + 1 * np.random.rand(m,1)

#%%
#Procesamiento
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_poly)

model = LinearRegression()
model.fit(x_scaled,y)
print('R2: ', model.score(x_scaled,y))

#%%
#Dibujo - - - - - - - - - - - - - - - - - -
x_new = np.linspace(-3,3,).reshape(-1,1)
x_new_poly = poly.transform(x_new)
x_new_scaled = scaler.transform(x_new_poly)
y_new = model.predict(x_new_scaled)

plt.plot(x_new,y_new,'r-',linewidth=2)
plt.plot(x,y,'b.')
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$y_1$', fontsize=18)
plt.axis([-3,3,0,10])
plt.show()
