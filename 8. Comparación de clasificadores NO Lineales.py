#%%
#Importaciones y Lectura de DataSet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.datasets import make_classification

#Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Crear modelos
classifiers = {'MLP': MLPClassifier(alpha=1, max_iter=1000), 
               'KNN': KNeighborsClassifier(3),
               'SVC': SVC(gamma=2, C=1),
               'DT':  DecisionTreeClassifier(max_depth=5),
               'GNB': GaussianNB()}
#Datasets
x, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1)
rng = np.random.RandomState(2)
x += 1 * rng.uniform(size=x.shape)
linearly_separable = (x,y)

datasets = [make_moons(noise=0.1),
            make_circles(noise=0.1, factor=0.5),
            linearly_separable]
#Color
from matplotlib.colors import ListedColormap
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#Probar modelo
figure = plt.figure(figsize=(9,3))
h = 0.02 #step


model_name = 'GNB' #Cambiar el nombre dependiendo el modelo a usar

#%%
for ds_count, ds in enumerate(datasets):
    x, y = ds
    x = StandardScaler().fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x,y)
    #para el dibujo
    xmin, xmax = x[:,0].min()-0.5, x[:,0].max()+0.5
    ymin, ymax = x[:,1].min()-0.5, x[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h ),
                         np.arange(ymin, ymax, h ))
    #----------------------------------
    
    model = classifiers[model_name]
    model.fit(xtrain, ytrain)
    score_train = model.score(xtrain, ytrain)
    score_test = model.score(xtest, ytest)
    
    #seguir dibujo
    ax = plt.subplot(1, 3, ds_count+1)
    
    if hasattr(model, 'decision_function'):
        zz = model.decision_function(np.c_[xx.ravel(),
                                           yy.ravel()])
    else:
        zz = model.predict_proba(np.c_[xx.ravel(),
                                       yy.ravel()])[:,1]
    zz = zz.reshape(xx.shape)
    ax.contourf(xx,yy,zz, cmap=cm, alpha=0.8)
    
    #dibujar puntos
    ax.scatter(xtrain[:,0], xtrain[:,1],
               c=ytrain, cmap=cm_bright, edgecolors='k')
    ax.scatter(xtest[:,0], xtest[:,1],
               c=ytest, cmap=cm_bright, edgecolors='k', alpha=0.6)
    
    
    #Dibujar puntos
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.text(xmax-0.3, ymin+0.7, '%.2f' %score_train,
            size=15, horizontalalignment='right')
    ax.text(xmax-0.3, ymin+0.7, '%.2f' %score_test,
            size=15, horizontalalignment='right')
plt.suptitle('Classification test: '+ model_name)
    
plt.tight_layout()
plt.show()


# %%
