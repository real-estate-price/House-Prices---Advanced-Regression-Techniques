import pandas

'''Код переносит столбец цен на 1-е место'''
data = pandas.read_csv('new_new_data.csv')
print(data.shape)
data.iloc[:, [1, 304]] = data.iloc[:, [304, 1]]
data.rename(columns={'MSSubClass': 'SalePriceForFeet', 'SalePriceForFeet': 'MSSubClass'}, inplace=True)
data.to_csv('new_data.csv', index=False)