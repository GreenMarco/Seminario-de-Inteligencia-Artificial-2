#%%
#Importaciones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


#%%
#Lectura de Datos

data = pd.read_csv('DataSets/mnist_784.csv')

data = pd.read_csv('csv/mnist_784.csv')


n_samples = 10000

x = np.asanyarray(data.drop(columns='class'))[:n_samples,:]
y = np.asanyarray(data[['class']])[:n_samples].ravel()



#%%
#Dibujar numeros random
"""sample = np.random.randint(n_samples)
plt.imshow((x[sample].reshape(28,28)), cmap=plt.cm.gray)
plt.tittle('Target: %i'%y[sample])
plt.show()"""



#%%
#Entrenamiento
xtrain,xtest,ytrain,ytest = train_test_split(x,y)

model = Pipeline([('scaler', StandardScaler()),
                  ('PCA', PCA(n_components=50)),
                  ('SVM', SVC(gamma=0.0001))])

model.fit(xtrain, ytrain.ravel())
print('Train: ', model.score(xtrain,ytrain))
print('Test: ', model.score(xtest,ytest))

ypred= model.predict(xtest)
print('\n\n Clasification matrix \n\n',confusion_matrix(ytest, ypred))
print('\n\n Clasification report \n\n', classification_report(ytest, ypred))




#%%
#Dibujar
sample = np.random.randint(xtest.shape[0])
plt.imshow((xtest[sample].reshape(28,28)), cmap=plt.cm.gray)
plt.title('Prediction: %i' %ypred[sample])
plt.show()



#%%
#Guardar En un archivo
#en una hoja BLANCA escribir un caracter para hacer un procesamiento para que quede de 28x28
import pickle
pickle.dump(model, (open('mnist_classifier.sav','wb')))
