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

Ошибка: 12,7%
"""

file1 = pd.read_csv('train (2).csv', index_col=0)
file2 = pd.read_csv('test.csv', index_col=0)

file1['LotShape'] = file1['LotShape'].replace(['Reg', 'IR1', 'IR2', 'IR3'], [4, 3, 2, 1]).astype('float')
file1['Utilities'] = file1['Utilities'].replace(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], [4, 3, 2, 1]).astype('float')
file1['LotConfig'] = file1['LotConfig'].replace(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], [2, 3, 1, 4, 5]).astype('float')
file1['LandSlope'] = file1['LandSlope'].replace(['Gtl', 'Mod', 'Sev'], [3, 2, 1]).astype('float')
file1['BldgType'] = file1['BldgType'].replace(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], [5, 3, 4, 2, 1]).astype('float')
file1['ExterQual'] = file1['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['ExterCond'] = file1['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['BsmtQual'] = file1['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['BsmtCond'] = file1['BsmtCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['BsmtExposure'] = file1['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No'], [4, 3, 2, 1]).astype('float')
file1['HeatingQC'] = file1['HeatingQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['Electrical'] = file1['Electrical'].replace(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], [5, 4, 3, 2, 3]).astype('float')
file1['KitchenQual'] = file1['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['FireplaceQu'] = file1['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['GarageFinish'] = file1['GarageFinish'].replace(['Fin', 'RFn', 'Unf'], [3, 2, 1]).astype('float')
file1['GarageQual'] = file1['GarageQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['GarageCond'] = file1['GarageCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file1['PavedDrive'] = file1['PavedDrive'].replace(['Y', 'P', 'N'], [3, 2, 1]).astype('float')
file1['MSSubClass'] = file1['MSSubClass'].astype('object')

file2['LotShape'] = file2['LotShape'].replace(['Reg', 'IR1', 'IR2', 'IR3'], [4, 3, 2, 1]).astype('float')
file2['Utilities'] = file2['Utilities'].replace(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], [4, 3, 2, 1]).astype('float')
file2['LotConfig'] = file2['LotConfig'].replace(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], [2, 3, 1, 4, 5]).astype('float')
file2['LandSlope'] = file2['LandSlope'].replace(['Gtl', 'Mod', 'Sev'], [3, 2, 1]).astype('float')
file2['BldgType'] = file2['BldgType'].replace(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], [5, 3, 4, 2, 1]).astype('float')
file2['ExterQual'] = file2['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['ExterCond'] = file2['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['BsmtQual'] = file2['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['BsmtCond'] = file2['BsmtCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['BsmtExposure'] = file2['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No'], [4, 3, 2, 1]).astype('float')
file2['HeatingQC'] = file2['HeatingQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['Electrical'] = file2['Electrical'].replace(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], [5, 4, 3, 2, 3]).astype('float')
file2['KitchenQual'] = file2['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['FireplaceQu'] = file2['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['GarageFinish'] = file2['GarageFinish'].replace(['Fin', 'RFn', 'Unf'], [3, 2, 1]).astype('float')
file2['GarageQual'] = file2['GarageQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['GarageCond'] = file2['GarageCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1]).astype('float')
file2['PavedDrive'] = file2['PavedDrive'].replace(['Y', 'P', 'N'], [3, 2, 1]).astype('float')
file2['MSSubClass'] = file2['MSSubClass'].astype('object')

y_train = file1.SalePrice
X_train = file1.drop(['SalePrice', 'PoolArea',
                      'LowQualFinSF', 'MoSold', 'Condition2', 'LotConfig', 'YrSold', 'MiscVal',
                      'LotFrontage', 'SaleType', 'BsmtHalfBath', 'BsmtFinSF2', 'ExterCond', 'ScreenPorch'], axis=1)
X_test = file2.drop(['PoolArea',
                     'LowQualFinSF', 'MoSold', 'Condition2', 'LotConfig', 'YrSold', 'MiscVal',
                     'LotFrontage', 'SaleType', 'BsmtHalfBath', 'BsmtFinSF2', 'ExterCond', 'ScreenPorch'], axis=1)

"""Выделяем категориальные и числовые данные"""
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]


for column in X_train.columns:
                if column == 'FireplaceQu':
                    X_train[column] = X_train[column].fillna(0)
                    X_test[column] = X_test[column].fillna(0)
                elif column == 'Alley':
                    X_train[column] = X_train[column].fillna('No Alley')
                    X_test[column] = X_test[column].fillna('No Alley')
                elif column == 'PoolQC':
                    X_train[column] = X_train[column].fillna('No PoolQC')
                    X_test[column] = X_test[column].fillna('No PoolQC')
                elif column == 'Fence':
                    X_train[column] = X_train[column].fillna('No Fence')
                    X_test[column] = X_test[column].fillna('No Fence')
                elif column == 'MiscFeature':
                    X_train[column] = X_train[column].fillna('No MiscFeature')
                    X_test[column] = X_test[column].fillna('No MiscFeature')
                elif column == 'GarageYrBlt':
                    X_train[column] = X_train[column].fillna(0)
                    X_test[column] = X_test[column].fillna(0)
                elif column in ('GarageCond', 'GarageFinish', 'GarageQual'):
                    X_train[column] = X_train[column].fillna(0)
                    X_test[column] = X_test[column].fillna(0)
                elif column == 'GarageType':
                    X_train[column] = X_train[column].fillna('No')
                    X_test[column] = X_test[column].fillna('No')
                elif column in ('BsmtExposure', 'BsmtQual', 'BsmtCond'):
                    X_train[column] = X_train[column].fillna(0)
                    X_test[column] = X_test[column].fillna(0)
                elif column == 'BsmtFinType1':
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

model = CatBoostRegressor(learning_rate=0.060532650918832094, max_depth=5, colsample_bylevel=0.18427044618770383,
                          boosting_type='Plain', bootstrap_type='Bernoulli', subsample=0.2772739302987132, loss_function='RMSE', iterations=1000,
                          verbose=0, random_state=42)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])
my_pipeline.fit(X_train, y_train)
ids = pd.DataFrame([i for i in range(1461, 2920)], columns=['Id'])
ids['SalePrice'] = pd.DataFrame(my_pipeline.predict(X_test), columns=['SalePrice'])
pd.DataFrame.to_csv(ids, 'submission.csv', index=False)
