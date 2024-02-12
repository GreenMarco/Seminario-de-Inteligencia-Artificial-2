import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('DataSets/daily-min-temperatures.csv')

plt.figure()
data.plot(y='Temp')

#x = np.asanyarray(data[['Temp']])


#Grafico de dispersion con retardo
#delay = 154
#plt.figure()
#Todos menos el primero y todos menos el ultimo
#plt.scatter(x[delay:], x[:-delay])
#print('Corr: ',np.corrcoef(x[delay:].T,x[:-delay].T)[0,1])

#grafica de autocorrelacion
pd.plotting.autocorrelation_plot(data.Temp)


df= pd.DataFrame(data.Temp)

delay = 5

for i in range (1, delay+1):
    df= pd.concat([df, data.Temp.shift(-i)], axis=1)
    
df = df[:-delay]

x = np.asanyarray(df.iloc[:,:-1])
y = np.asanyarray(df.iloc[:,-1])

xtrain, xtest,ytrain, ytest, = train_test_split(x,y)

model = LinearRegression()
model.fit(xtrain, ytrain)

print('Train:', model.score(xtrain, ytrain))
print('Test:', model.score(xtest, ytest))