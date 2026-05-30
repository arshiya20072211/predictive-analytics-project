import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample historical sales data
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [100, 120, 130, 150, 170, 180, 200, 220, 240, 260]
}

df = pd.DataFrame(data)

# Training data
X = df[['Month']]
y = df['Sales']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_months = pd.DataFrame({'Month': [11, 12, 13]})
predictions = model.predict(future_months)

print("Predicted Sales:")
for month, sale in zip(future_months['Month'], predictions):
    print(f"Month {month}: {sale:.2f}")

# Visualization
plt.scatter(df['Month'], df['Sales'])
plt.plot(df['Month'], model.predict(X))
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting Using Linear Regression")
plt.show()
