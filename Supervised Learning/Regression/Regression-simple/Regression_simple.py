# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:55:21 2019

@author: joke
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:37:48 2018

"""
#importation librairy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbrn
import statsmodels as stat
from scipy import stats
from math import sqrt

#creation dataset
dtst =  pd.read_csv('reg_simple.csv')
 
X1 = dtst.iloc[:,:-1].values
Y1 = dtst.iloc[:,-1].values



plt.scatter(X1,Y1)
plt.xlabel('heure_rev_independante_var')
plt.ylabel('note : dependante variable')
plt.style.use(['dark_background','fast'])
plt.show()




 # Construction de l'echantillon de trainning et de l'echantillon de test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(X1,Y1, test_size = 0.2, random_state = 0)
 
 #construction du model  de regression simple
from sklearn.linear_model import LinearRegression
regresseur = LinearRegression()
regresseur.fit(X_train,y_train)
 
 #etablir une prediction
y_prediction =  regresseur.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regresseur.predict(X_train), color = 'blue')
plt.xlabel('heure_rev_independante_var')
plt.ylabel('dependante variable - NOTE')
plt.title('droite de gression')
plt.style.use(['dark_background','fast'])
plt.show()

test_X = np.array(18).reshape(-1, 1)
regresseur.predict(test_X)


#racine carrée du  carré moyen des erreurs ou racine carrée de erreur quadratique moyenne
#c’est la moyenne arithmétique des carrés des écarts entre les prévisions et les observations.
def rmse(z,z_hat):
    y_actual=np.array(z).reshape(-1, 1)#  les observations
    y_pred=np.array(z_hat) #les prévisions
    error=(y_actual-y_pred)**2 # carrés des écarts entre les prévisions et les observations
    error_mean=round(np.mean(error)) # carré moyen des erreurs ou erreur quadratique moyenne
    err_sq=sqrt(error_mean) #  racine carrée du carré moyen des erreurs ou erreur quadratique moyenne
    return err_sq

rmse(66,regresseur.predict(np.array(18).reshape(-1, 1)))