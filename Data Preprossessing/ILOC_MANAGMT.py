# Sélections uniques avec iloc et DataFrame
# Lignes:
data.iloc [ 0 ] # première ligne du bloc de données 
data.iloc [ 1 ] # deuxième rangée de trame de données 
data.iloc [ - 1 ] # dernière ligne du bloc de données 
# Colonnes:
data.iloc [:, 0 ] # première colonne de trame de données 
data.iloc [:, 1 ] # deuxième colonne de trame de données 
data.iloc [:, - 1 ] # dernière colonne de trame de données 
# Sélections multiples de lignes et de colonnes à l'aide de iloc et DataFrame
data.iloc [ 0 : 5 ] # les cinq premières lignes du cadre de données
data.iloc [:, 0 : 2 ] # les deux premières colonnes du cadre de données contenant toutes les lignes.
data.iloc [[ 0 , 3 , 6 , 24 ], [ 0 , 5 , 6 ]] # 1ère, 4ème, 7ème, 25ème rangée + 1ère 6ème 7ème colonne.
data.iloc [ 0 : 5 , 5 : 8 ] # 5 premières lignes et 5ème, 6ème, 7ème colonnes de la trame de données 