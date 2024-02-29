#%%
#Importaciones y Lectura de DataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

data = pd.read_csv("DataSets/diabetes.csv")


#%%
#Correlacion
corr = data.corr()
import seaborn as sns
sns.heatmap(corr,
                xticklabels= corr.columns,
                yticklabels= corr.columns
            )


#%%
#Regresión Logística
x = np.asanyarray(data[["Glucose"]])
y = np.asanyarray(data[["Outcome"]])

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = LogisticRegression(solver= "lbfgs")
model.fit(xtrain, ytrain)

print("train: ", model.score(xtrain,ytrain))
print("test: ", model.score(xtest, ytest))

#Dibujo 
g= np.linspace(0, 200,80). reshape(-1,1)
#si quiero una prediccion, uso el proba
pred = model.predict_proba(g)
plt.plot(xtrain, ytrain, ".b")
plt.plot(xtest,ytest, ".r")
plt.xlabel("Glucose")
plt.ylabel("Outcome")
plt.plot(g, pred, "-k")




#%%
#Modelos mas grandes

# seleccion de variables
x = np.asanyarray(data.drop(columns = ["Glucose"]))
y = np.asanyarray(data[["Outcome"]])

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

#Crear modelo
model = Pipeline([("scaler", StandardScaler()),
                  ("logit", LogisticRegression())])

#Entrenar
model.fit(xtrain, ytrain.ravel())
print("train: ", model.score(xtrain,ytrain))
print("test: ", model.score(xtest, ytest))

# Explicacion de variables
coef = np.abs(model.named_steps["logit"].coef_[0])
labels = list(data.drop(columns=["Outcome"]).columns)
features = pd.DataFrame()
features["Features"] = labels
features["Importance"] = coef/np.sum(coef)
features.sort_values(by=["Importance"], ascending=True,
                     inplace=True)
features.set_index("Features", inplace=True)
features.Importance.plot(kind="barh")
plt.xlabel("Importance")



# %%
#Iris
iris = load_iris()

x = iris["data"][:, (2,3)]
y = iris["target"]

plt.plot(x[y== 2, 0], x[y==2, 1], 'g^', label = "Virginica")
plt.plot(x[y== 1, 0], x[y==1, 1], 'bs', label = "Versicolor")
plt.plot(x[y== 0, 0], x[y==0, 1], 'yo', label = "Virginica")
xtrain, xtest, ytrain, ytest = train_test_split(x,y)
softmax_reg = LogisticRegression(multi_class= "multinomial",
                                 solver="lbfgs", C=10)
softmax_reg.fit(xtrain, ytrain)
print("train: ", softmax_reg.score(xtrain,ytrain))
print("test: ", softmax_reg.score(xtest, ytest))
