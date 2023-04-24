from sklearn.ensemble import RandomForestRegressor
from Models.data_reader import make_data
import numpy as np

def main():
    X_train, X_test, y_train, y_test = make_data('train one-hot encoding.csv')
    
    mod = RandomForestRegressor(n_estimators=100, random_state=0)
    mod.fit(X_train, y_train)
    prediction_2 = np.array(list(map(round, mod.predict(X_test))))
    print((np.linalg.norm(prediction_2 - y_test)) / np.linalg.norm(y_test))
    
if __name__ == '__main__':
    main()
