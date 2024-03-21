import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Data.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop(['status'], axis=1)  # Features
y = df['status']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train_scaled, y_train)

# Predict on the testing data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_probs = model.predict_proba(X_test_scaled)

# Plot the predicted probabilities
plt.figure(figsize=(10, 6))
sns.histplot(y_probs, bins=50, kde=True)
plt.title('Predicted Probabilities for Each Class')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend(['Class 0', 'Class 1'])
plt.show()