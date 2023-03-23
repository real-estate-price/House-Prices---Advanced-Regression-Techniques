from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv


def main():
    reader = csv.reader(open("new_data.csv", "r"), delimiter=",")
    S = list(reader)[1:]
    D = [list(map(int, x)) for x in S]
    A = np.array(D)[:, 1:]
    y = A[:, 0]
    A = A[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(
        A, y, random_state=0)
    mod = LinearRegression()
    mod.fit(X_train, y_train)
    prediction = mod.predict(X_test)
    print(prediction)
    print(np.linalg.norm(prediction - y_test), np.linalg.norm(y_test))
    print((np.linalg.norm(prediction - y_test)) / np.linalg.norm(y_test))


if __name__ == '__main__':
    main()
