import csv

import pandas
from pandas import *

'''Код создаёт столбец цен на квадратный метр и удаляет старый столбец цен и столбец площади'''
data = read_csv('new train.csv')

area = data['LotArea'].tolist()
prices = data['SalePrice'].tolist()
for i in range(len(area)):
    prices[i] = int(prices[i]) / int(area[i])
data['SalePriceForFeet'] = pandas.Series(prices)
data.pop('SalePrice')
data.pop('LotArea')
data.to_csv('new_new_data.csv', index=False)