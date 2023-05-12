import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
"""Здесь используется модель RandomForest. Предобработка данных: NaN заменятеся на наиболее частое значение в столбце,
а категориальные значения обрабатываются с помощью onehot.
Ошибка: 16,8%
Замечания: надо бы поиграться с параметрами randomforest 
"""



file = pd.read_csv('train (2).csv', index_col=0)
y = file.SalePrice
X = file.drop(['SalePrice'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y,  train_size=0.8, test_size=0.2, random_state=0)

"""Выделяем категориальные и числовые данные"""
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

"""1-й шаг pipelies"""
numerical_transformer = SimpleImputer(strategy='constant')  # для числовых данных

categorical_transformer = Pipeline(steps=[  # для категориальных
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(  # применяем ко всем столбцам
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])


model = RandomForestRegressor(n_estimators=500, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
prediction = my_pipeline.predict(X_valid)
print((np.linalg.norm(prediction - y_valid)) / np.linalg.norm(y_valid))
