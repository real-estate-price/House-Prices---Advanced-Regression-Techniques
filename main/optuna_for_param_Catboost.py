y_trans = np.log(y_train)


def objective(trial):
    params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
              'max_depth': trial.suggest_int('max_depth', 2, 12),
              'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1),
              'boosting_type': trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
              'bootstrap_type': trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
              'used_ram_limit': "3gb"}

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cb = CatBoostRegressor(**params, loss_function='RMSE', iterations=1000, verbose=0, random_state=42)
    return np.sqrt(-1 * cross_val_score(cb, X_train, y_trans, scoring='neg_mean_squared_error').mean())
