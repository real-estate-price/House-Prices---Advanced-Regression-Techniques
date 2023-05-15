import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

"""Здесь используется модель XGBRegressor.

Предобработка данных: отранжировали признаки, которые можно было; 
 удалили 4 признака, где слишком много пропущенных данных(>80%); удалили некоторые признаки, 
 которые по анализу Борута считались неважными(брали чужой;!несколько из них оказались важными и повышали точность);
 NaN заменятеся на наиболее частое значение в столбце, а категориальные значения обрабатываются с помощью onehot.

Ошибка: 14%

Замечания: ручной подбор параметров модели ухудшает точность(даже с помощью 
неручного(хотя перебирал только значения n_estimators)). Поэтому дэфолтные значения вроде как самые оптимальные
"""

file = pd.read_csv('train (2).csv', index_col=0)
y = file.SalePrice
file['LotShape'] = file['LotShape'].replace(['Reg', 'IR1', 'IR2', 'IR3'], [4, 3, 2, 1])
file['Utilities'] = file['Utilities'].replace(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], [4, 3, 2, 1])
file['LotConfig'] = file['LotConfig'].replace(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], [2, 3, 1, 4, 5])
file['LandSlope'] = file['LandSlope'].replace(['Gtl', 'Mod', 'Sev'], [3, 2, 1])
file['BldgType'] = file['BldgType'].replace(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], [5, 3, 4, 2, 1])
file['ExterQual'] = file['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['ExterCond'] = file['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['BsmtQual'] = file['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['BsmtCond'] = file['BsmtCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['BsmtExposure'] = file['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No'], [4, 3, 2, 1])
file['HeatingQC'] = file['HeatingQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['Electrical'] = file['Electrical'].replace(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], [5, 4, 3, 2, 3])
file['KitchenQual'] = file['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['FireplaceQu'] = file['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['GarageFinish'] = file['GarageFinish'].replace(['Fin', 'RFn', 'Unf'], [3, 2, 1])
file['GarageQual'] = file['GarageQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['GarageCond'] = file['GarageCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [5, 4, 3, 2, 1])
file['PavedDrive'] = file['PavedDrive'].replace(['Y', 'P', 'N'], [3, 2, 1])
file['PoolQC'] = file['PoolQC'].replace(['Ex', 'Gd', 'TA', 'Fa'], [5, 4, 3, 2])


X = file.drop(['SalePrice', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'PoolArea',
               'LowQualFinSF', 'MoSold', 'Condition2', 'LotConfig', 'YrSold', 'MiscVal',
               'LotFrontage', 'SaleType', 'BsmtHalfBath', 'BsmtFinSF2', 'ExterCond', 'ScreenPorch'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

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

model = XGBRegressor()
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])
my_pipeline.fit(X_train, y_train)
prediction = my_pipeline.predict(X_valid)
print((np.linalg.norm(prediction - y_valid)) / np.linalg.norm(y_valid))
