# Importing dependencies
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Custom_Logistic_Regression import LogisticRegression

X, y = load_breast_cancer(return_X_y = True)
print(X.shape)
print(y.shape)

# Visualizing the data
plt.scatter(x = X[:, 0], y = X[:, 1], cmap = 'copper', c = y)
plt.show()

# Splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 21)

# Training and evaluating the model
model = LogisticRegression()

model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print(classification_report(y_test, test_pred))
