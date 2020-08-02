# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:48:22 2019

@author: joke
"""

import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#importe LE FICHIER CSV
df=pd.read_csv('age.csv')

df = df.drop(['Sexe'], axis = 1)

x=df.iloc[:,:-1].values

y=df['Age'].values

regressor=SVR(kernel='linear',degree=1)

#TRACER LA RELATION:

plt.scatter(df['Poids écaillé'],df['Age'])

xtrain,xtest,ytrain,ytest=train_test_split(x,y)
regressor.fit(xtrain,ytrain)


pred=regressor.predict(xtest)


#EVALUER LE MODEL ET COMPARER
#VÉRIFIER L'EXACTITUDE
#Note: .score ()  nous donnent le score de précision de la prédiction
print(regressor.score(xtest,ytest))
print(r2_score(ytest,pred))



regressor =SVR(kernel='rbf',epsilon=1.0)
#ici nous plaçons le noyau sur 'rbf' du degré 3 et sur une valeur epsilon de 1,0 (Noyau de type Gaussien)
#Les autres noyaux sont → 'linéaire', 'poly' (pour polynôme), 'rbf'

regressor.fit(xtrain,ytrain)
pred_2=regressor.predict(xtest)


print(regressor.score(xtest,ytest))
print(r2_score(ytest,pred_2))




