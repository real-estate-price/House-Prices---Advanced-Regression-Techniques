import csv
import numpy as np
from sklearn.model_selection import train_test_split


def change(x:list):  # функция заменяет пустые элементы листа на 0
    for i in range(len(x)):
        if x[i] == '':
            x[i] = 0
    return x


def make_data(file_name):
    reader = csv.reader(open(file_name, "r"), delimiter=",")
    A = list(reader)[1:]  # удаляем названия столбцов
    A = [list(map(float, change(x))) for x in A]  # заменяем пустые элементы таблицы на 0 и приводим к float
    A = np.array(A)[:, 1:]  # удаляем столбец с id
    y = A[:, 0]  # вектор фактических цен(находится в столбце 37)
    A = A[:, 1:]  # матрица признаков
    X_train, X_test, y_train, y_test = train_test_split(
        A, y, random_state=0)
    return X_train, X_test, y_train, y_test
