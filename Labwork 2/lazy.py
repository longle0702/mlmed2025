import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_percentage_error

df = pd.read_csv(r'training_set_pixel_size_and_HC.csv')
X = df[['pixel size(mm)']]
y = df['head circumference (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_absolute_error)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)