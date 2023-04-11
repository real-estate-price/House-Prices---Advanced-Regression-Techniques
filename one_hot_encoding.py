import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# нужно создать датафрейм
df = pd.read_csv('test.csv', encoding="utf-8", delimiter=',')
# разбиваем на числовые и категориальные признаки
X_num = df.select_dtypes(exclude='object')
X_cat = df.select_dtypes(include='object')
# невидимые категории кодирует 0 (ignore)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_cat)
# Чтобы новые признаки назывались красиво (как старые и через _ значение)
categorical_columns = [f'{col}_{cat}' for i, col in enumerate(X_cat.columns) for cat in encoder.categories_[i]]
# соединяем полученный с исходным
one_hot_features = pd.DataFrame(X_encoded, columns=categorical_columns)
df = X_num.join(one_hot_features)