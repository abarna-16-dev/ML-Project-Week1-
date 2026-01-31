import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Load diamonds dataset
datasets = sns.load_dataset('diamonds')
datasets.to_csv('diamonds.csv', index=False)

# Read dataset
data = pd.read_csv('diamonds.csv')
df = pd.DataFrame(data)

# Feature and target
X = df[['carat']]        # independent variable
y = df['price']          # dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation: Price = {m:.2f} × Carat + {c:.2f}")

# Predict for user input
user_carat = float(input("Enter carat value: "))
user_carat_df = pd.DataFrame({'carat': [user_carat]})

predicted_price = model.predict(user_carat_df)
print(f"Predicted diamond price: {predicted_price[0]:.2f}")