import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Assuming df is your dataframe
# Assuming 'sex' and 'smoker' are categorical variables
df = pd.get_dummies(df, columns=['sex', 'smoker'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('expenses', axis=1)
y = df['expenses']

# Split the data into training and testing sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(train_dataset, train_labels)
# Make predictions on the test dataset
predictions = model.predict(test_dataset)

# Evaluate the model
mae = mean_absolute_error(test_labels, predictions)
print("Mean Absolute Error:", mae)
# Make predictions on the test dataset for visualization
test_predictions = model.predict(test_dataset)

# Plot actual vs predicted expenses
import matplotlib.pyplot as plt
plt.scatter(test_labels, test_predictions)
plt.xlabel('Actual Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Expenses')
plt.show()
