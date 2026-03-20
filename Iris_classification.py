from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Data loading
iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()

#Model training
model.fit(X_train, y_train)

#Prediction
y_predict = model.predict(X_test)

#Checking model accuracy
accuray = accuracy_score(y_test, y_predict)
print("accuracy:", accuray)
