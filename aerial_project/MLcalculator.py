import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Dataset: [a, b, operation]
# operation: 0=+, 1=-, 2=*, 3=/
X = np.array([
    [2, 1, 0], [5, 3, 0],   # addition
    [5, 2, 1], [9, 4, 1],   # subtraction
    [2, 3, 2], [4, 5, 2],   # multiplication
    [6, 2, 3], [8, 4, 3]    # division
])

y = np.array([
    3, 8,
    3, 5,
    6, 20,
    3, 2
])

# Train model
model = DecisionTreeRegressor()
model.fit(X, y)

# Input
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

print("Select operation:")
print("0 = +, 1 = -, 2 = *, 3 = /")
op = int(input("Enter choice: "))

result = model.predict([[a, b, op]])

print("Result:", result[0])