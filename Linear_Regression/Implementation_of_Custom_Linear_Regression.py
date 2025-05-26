# Importing dependencies
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from Custom_Linear_Regression import LinearRegression

X, y = make_regression(n_samples = 200, n_features = 1, noise = 10, random_state = 21)
print(X.shape)
print(y.shape)

# Visualizing the data
plt.scatter(x = X[:, 0], y = y, s = 30)
plt.show()

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 21)

# Training and Evaluating the model
model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
train_acc = mean_squared_error(y_train, train_pred)

test_pred = model.predict(X_test)
test_acc = mean_squared_error(y_test, test_pred)

print("Accuracy on training data : ", train_acc)
print("Accuracy on testing data : ", test_acc)