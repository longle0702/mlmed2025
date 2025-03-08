from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

df = pd.read_csv(r'/home/long/longdata/mlmed/prac2/1/training_set_pixel_size_and_HC.csv')
X = df[['pixel size(mm)']]
y = df['head circumference (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor()
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 150, 200],
    'max_depth': [None, 2, 3],
    'min_samples_leaf': [10, 15, 20]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Best MAE:", mae)

joblib.dump(best_model, 'best_hgbr.pkl')
