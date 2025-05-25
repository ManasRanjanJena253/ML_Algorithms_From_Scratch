# Importing dependencies
from Custom_KNN import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Loading the data
X, y = load_iris(return_X_y = True)

# Visualizing the data
plt.scatter(x = X[:, 0], y = X[:, 1], c = y, cmap = 'spring')
plt.show()

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state = 21, stratify = y, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)

# Model training and testing
model = KNN(k = 3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)

print(report)