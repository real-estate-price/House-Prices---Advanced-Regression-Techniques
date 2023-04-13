from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv


def main():
    reader = csv.reader(open("test_and_train_one-hotencoding.csv", "r"), delimiter=",")
    A = list(reader)[1:]  # удаляем названия столбцов
    A = [list(map(float, x)) for x in A]
    A = np.array(A)[:, 1:]  # удаляем столбец с id
    y = A[:, 0]  # вектор фактических цен
    A = A[:, 1:]  # матрица признаков
    X_train, X_test, y_train, y_test = train_test_split(
        A, y, random_state=0)  # делим выборку на train и test

    '''Обычная линейная регрессия(ошибка очень большая)'''
    mod = LinearRegression()
    mod.fit(X_train, y_train)
    prediction = mod.predict(X_test)
    print(prediction)

    '''L1 регуляризация(ошибка 20%)'''
    # mod_2 = Lasso(alpha=0.5)
    # mod_2.fit(X_train, y_train)
    # prediction_2 = np.array(list(map(round, mod_2.predict(X_test))))
    # print((np.linalg.norm(prediction_2 - y_test)) / np.linalg.norm(y_test))

    '''L2 регуляризация(ошибка 17%)'''
    # mod_3 = Ridge(alpha=15)
    # mod_3.fit(X_train, y_train)
    # prediction_3 = np.array(list(map(round, mod_3.predict(X_test))))
    # print((np.linalg.norm(prediction_3 - y_test)) / np.linalg.norm(y_test))

    '''если брать среднюю цену, ошибка 40%'''
    # pred_4 = np.array([sum(list(y_test)) / len(list(y_test))] * len(list(y_test)))


if __name__ == '__main__':
    main()
