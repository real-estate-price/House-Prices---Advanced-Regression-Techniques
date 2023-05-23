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

"""Здесь используется модель CatBoostRegressor.

Предобработка данных: удалили 4 признака, где слишком много пропущенных данных(>80%); удалили некоторые признаки, 
которые по анализу Борута считались неважными; в то столбцах, где NAN скорее всего означает отсутствие, NAN заменили
на No... В остальных столбцах NaN заменятеся на наиболее частое значение в столбце;
категориальные значения обрабатываются с помощью onehot; для подбора гипер параметров использовали optuna.

Ошибка: 13,2%
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


for column in X_train.columns:
                if column == 'FireplaceQu':
                    X_train[column] = X_train[column].fillna('No Fireplace')
                    X_test[column] = X_test[column].fillna('No Fireplace')
                elif column == 'GarageYrBlt': 
                    X_train[column] = X_train[column].fillna(0)
                    X_test[column] = X_test[column].fillna(0)
                elif column in ('GarageCond', 'GarageType', 'GarageFinish', 'GarageQual'):
                    X_train[column] = X_train[column].fillna('No Garage')
                    X_test[column] = X_test[column].fillna('No Garage')
                elif column in ('BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1'):
                    X_train[column] = X_train[column].fillna('No Basement')
                    X_test[column] = X_test[column].fillna('No Basement')
                    

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

model = CatBoostRegressor(learning_rate=0.14144651033395983, max_depth=3, colsample_bylevel=0.7573342483166768,
                          boosting_type='Ordered', bootstrap_type='MVS', loss_function='RMSE', iterations=1000,
                          verbose=0, random_state=42)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])
my_pipeline.fit(X_train, y_train)
ids = pd.DataFrame([i for i in range(1461, 2920)], columns=['Id'])
ids['SalePrice'] = pd.DataFrame(my_pipeline.predict(X_test), columns=['SalePrice'])
pd.DataFrame.to_csv(ids, 'submission.csv', index=False)
