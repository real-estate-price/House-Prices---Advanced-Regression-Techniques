import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
"""Здесь используется модель XGBRegressor.

Предобработка данных: удалил 4 признака, где слишком много пропущенных данных(>80%); удалил некоторые признаки, 
 которые по анализу Борута считались неважными(брал чужой;!несколько из них оказались важными и повышали точность);
 NaN заменятеся на наиболее частое значение в столбце, а категориальные значения обрабатываются с помощью onehot.
 
Ошибка: 12,7%

Замечания: ручной подбор параметров модели ухудшает точность(даже с помощью 
неручного(хотя перебирал только значения n_estimators)). Поэтому дэфолтные значения вроде как самые оптимальные
"""


file1 = pd.read_csv('train (2).csv', index_col=0)
file2 = pd.read_csv('test.csv', index_col=0)
y_train = file1.SalePrice
X_train = file1.drop(['SalePrice', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'PoolArea',
               'LowQualFinSF', 'MoSold', 'Condition2', 'LotConfig', 'YrSold', 'MiscVal',
               'LotFrontage', 'SaleType', 'BsmtHalfBath', 'BsmtFinSF2', 'ExterCond', 'ScreenPorch'], axis=1)
X_test = file2.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'PoolArea',
               'LowQualFinSF', 'MoSold', 'Condition2', 'LotConfig', 'YrSold', 'MiscVal',
               'LotFrontage', 'SaleType', 'BsmtHalfBath', 'BsmtFinSF2', 'ExterCond', 'ScreenPorch'], axis=1)

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


model = CatBoostRegressor(learning_rate=0.14144651033395983, max_depth=3, colsample_bylevel=0.7573342483166768, boosting_type='Ordered', bootstrap_type='MVS', loss_function='RMSE', iterations=1000, verbose=0, random_state=42)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
my_pipeline.fit(X_train, y_train)
ids = pd.DataFrame([i for i in range(1461, 2920)], columns=['Id'])
ids['SalePrice'] = pd.DataFrame(my_pipeline.predict(X_test),columns=['SalePrice'])
pd.DataFrame.to_csv(ids, 'submission.csv', index=False)
