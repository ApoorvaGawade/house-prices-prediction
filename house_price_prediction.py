import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

data = pd.read_csv(r"C:\Users\Apoorva\Downloads\house_price_prediction_dataset\train.csv")

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

mae = mean_absolute_error(ytest, ypred)
rmse = root_mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print("MAE : ", mae)
print("RMSE : ", rmse)
print("R2 score : ", r2)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)

sample_house = [[2000, 3, 2]] 
predicted_price = model.predict(sample_house)
print("Predicted House Price:", predicted_price[0])

plt.scatter(ytest, ypred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()








