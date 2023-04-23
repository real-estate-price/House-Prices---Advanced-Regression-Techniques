from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import csv


def change(x:list):  # функция заменяет пустые элементы листа на 0
    for i in range(len(x)):
        if x[i] == '':
            x[i] = 0
    return x


def main():
    reader = csv.reader(open("train one-hot encoding.csv", "r"), delimiter=",")
    A = list(reader)[1:]  # удаляем названия столбцов
    A = [list(map(float, change(x))) for x in A]  # заменяем пустые элементы таблицы на 0 и приводим к float
    A = np.array(A)[:, 1:]  # удаляем столбец с id
    y = A[:, 36]  # вектор фактических цен(находится в столбце 37)
    A = np.hstack((A[:, :36], A[:, 37:]))  # матрица признаков
    X_train, X_test, y_train, y_test = train_test_split(
        A, y, random_state=0)
    
    mod_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    mod_2.fit(X_train, y_train)
    prediction_2 = np.array(list(map(round, mod_2.predict(X_test))))
    print((np.linalg.norm(prediction_2 - y_test)) / np.linalg.norm(y_test))
    
if __name__ == '__main__':
    main()
