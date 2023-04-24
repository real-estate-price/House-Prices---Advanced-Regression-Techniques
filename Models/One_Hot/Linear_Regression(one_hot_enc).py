from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from Models.data_reader import make_data


def main():
    X_train, X_test, y_train, y_test = make_data('train one-hot encoding.csv')

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
