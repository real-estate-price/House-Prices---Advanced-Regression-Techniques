def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
              'max_depth': trial.suggest_int('max_depth', 2, 20),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
              'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
              'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1)}

    xgb = XGBRegressor(**params, random_state=42)
    return np.sqrt(-1 * cross_val_score(xgb, X_train, y_train, scoring='neg_mean_squared_log_error').mean())


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(study.best_trial)
