from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import csv

def main():
    reader = csv.reader(open("new_data.csv  ", "r"), delimiter=",")
    A = list(reader)[1:]  # удаляем названия столбцов
    A = [list(map(float, x)) for x in A]
    A = np.array(A)[:, 1:]  # удаляем столбец с id
    y = A[:, 0]  # вектор фактических цен
    A = A[:, 1:]  # матрица признаков
    X_train, X_test, y_train, y_test = train_test_split(
        A, y, random_state=0)
    mod_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    mod_2.fit(X_train, y_train)
    prediction_2 = np.array(list(map(round, mod_2.predict(X_test))))
    print((np.linalg.norm(prediction_2 - y_test)) / np.linalg.norm(y_test))
