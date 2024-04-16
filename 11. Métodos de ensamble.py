#%%
#Importaciones
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

#%%
#Variables
data = pd.read_csv('DataSets/diabetes.csv')
x= np.asanyarray(data.iloc[:,:-1])
y= np.asanyarray(data.iloc[:,-1])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,
                                                random_state=0)


#%%
#Random Forest 
#Bagging de arbol de DTs con algunas heurisiticas adcionales
rf = RandomForestClassifier(n_estimators=100,)
rf.fit(xtrain, ytrain)

print("RF train: ", rf.score(xtrain,ytrain))
print("RF test: ", rf.score(xtest, ytest))

#%%
#Bagging
bg =  BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,
                        max_features=1.0,n_estimators=100)
bg.fit(xtrain, ytrain)

print("BG train: ", bg.score(xtrain,ytrain))
print("BG test: ", bg.score(xtest, ytest))


#%%
#Bagging
ab = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=5,
                        learning_rate=1)
ab.fit(xtrain, ytrain)

print("AB train: ", ab.score(xtrain,ytrain))
print("AB test: ", ab.score(xtest, ytest))



# %%
#Voting
lr = LogisticRegression(solver='lbfgs', max_iter=500)
dt = DecisionTreeClassifier()
svm= SVC(kernel='rbf', gamma='scale')

vt  =  VotingClassifier(estimators=[('lr',lr),('dt',dt), ('svm',svm)],
                        voting='hard')
vt.fit(xtrain, ytrain)

print("VT train: ", vt.score(xtrain,ytrain))
print("VT test: ", vt.score(xtest, ytest))
# %%
