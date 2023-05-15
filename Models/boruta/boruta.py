array_different_seeds = []
for seed_version in range(1, 100):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    # Data / new_data.csv
    X = pd.read_csv('new_data.csv', encoding="utf-8", delimiter=',')
    y = X['SalePrice']
    X.pop('SalePrice')
    # создание shadow features
    
    np.random.seed(seed_version)
    X_shadow = X.apply(np.random.permutation) 
    X_shadow.columns = ['shadow_' + feat for feat in X.columns] # returns names of the shandow dataframe if printed
    X_boruta = pd.concat([X,X_shadow], axis = 1)
  
    # create the rfr class object and specify max_depth, the number of trees used to make a prediction, as 5,
    # set an internal state of 42 so that can be used to generate pseudo numbers.
    forest = RandomForestRegressor(max_depth = 5, random_state=42)
    forest.fit(X_boruta, y) # uses the above specified internal state to fit model with 5 trees
    # важность признаков в X
    feat_imp_X = forest.feature_importances_[:len(X.columns)] 
    # важность признаков X_shandow
    feat_imp_shadow = forest.feature_importances_[len(X.columns):]
    # сравниваем важности в начальном и теневом
    hits = feat_imp_X > feat_imp_shadow.max()
    result= []
    for i in range(len(hits)):
        if hits[i]:
            array_different_seeds.append(X.columns[i])
from collections import Counter
# Признак и количество раз, когда он встречался как важный
Counter(array_different_seeds).most_common(len(X.columns))