from sklearn.ensemble import RandomForestRegressor
from Models.data_reader import make_data
import numpy as np


def main():
    X_train, X_test, y_train, y_test = make_data('new_data.csv')
    mod_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    mod_2.fit(X_train, y_train)
    prediction_2 = np.array(list(map(round, mod_2.predict(X_test))))
    print((np.linalg.norm(prediction_2 - y_test)) / np.linalg.norm(y_test))
