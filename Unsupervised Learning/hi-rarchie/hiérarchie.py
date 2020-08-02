# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:29:46 2019

@author: joke
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X = pd.read_csv('credit_bank.csv').values



from scipy.cluster.hierarchy import linkage, dendrogram

fusions = linkage(X, method='complete',metric='euclidean')

dendrogram(fusions,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.title('Dendrogram')
plt.xlabel('demandeurs de credit')
plt.ylabel('distances Euclidiennes')
plt.show()


# Adapter le clustering hiérarchique au jeu de données
from sklearn.cluster import AgglomerativeClustering
hierarchie = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
y_hierarchie = hierarchie.fit_predict(X)




plt.scatter(X[y_hierarchie == 0, 0], X[y_hierarchie == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_hierarchie == 1, 0], X[y_hierarchie == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(X[y_hierarchie == 2, 0], X[y_hierarchie == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters de demandeur_credit')
plt.xlabel('epargne en millier')
plt.ylabel('score_bank')
plt.legend()
plt.show()

















