# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. open the jupitor/Google collab
2.enter the code
3.run the program
4.hence the program successfully completed

   

## Program:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Copy dataset
data1 = data.copy()

# Drop unwanted columns
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Label Encoding
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Split input and output
x = data1.iloc[:, :-1].values
y = data1["status"].values

# Initialize theta
theta = np.random.randn(x.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient Descent
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)

    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient

    return theta

# Train model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Predict for training data
y_pred = predict(theta, x)

# Accuracy
accuracy = np.mean(y_pred.flatten() == y)

print("Accuracy:", accuracy)
print("Predicted:\n", y_pred)
print("Actual:\n", y)

# New data prediction
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])

y_prednew = predict(theta, xnew)

print("Predicted Result:", y_prednew)

Developed by: B.SASIREKHA
RegisterNumber: 212225040388

```

## Output:
<img width="834" height="347" alt="Screenshot 2026-08-24 195003" src="https://github.com/user-attachments/assets/cf7756b7-32a3-4df6-ae28-b4c7ec229fb7" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

