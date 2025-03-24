from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import joblib

df = pd.read_csv(r'D:\USTH\Nam Ba\ml in med\mlmed2025\Labwork 2\training_set_pixel_size_and_HC.csv')
X = df[['pixel size(mm)']]
y = df['head circumference (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 150],
    'min_samples_leaf': [1, 10],
    'loss' : ['absolute_error'],
    'warm_start' : [True, False],
    'max_leaf_nodes': [None, 10],
    'n_iter_no_change': [10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
print("MAPE:", mape)

joblib.dump(best_model, 'best_gbr.pkl')    