from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import joblib

df = pd.read_csv(r'/home/long/longdata/mlmed/prac2/1/training_set_pixel_size_and_HC.csv')
X = df[['pixel size(mm)']]
y = df['head circumference (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = HistGradientBoostingRegressor()
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 150, 200],
    'min_samples_leaf': [10, 15, 20],
    'l2_regularization': [0, 0.1, 0.001],
    'max_leaf_nodes': [11, 21, 31], 
    'warm_start' : [True, False],
    'loss' : ['absolute_error']
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

joblib.dump(best_model, 'best_hgbr.pkl')    