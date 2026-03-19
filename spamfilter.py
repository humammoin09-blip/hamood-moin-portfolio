import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


#loading data with the help of Pandas and doing lable encoding 
df = pd.read_csv(r"C:\Users\pc\Downloads\archive\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

#Defining X and y 
X = df['message']
y = df['label']

#Data splitting and using algorithem 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


model = MultinomialNB()

#Model training 
model.fit(X_train, y_train)

#Prediction
y_predict = model.predict(X_test)

#Checking model accuracy
accuracy = accuracy_score(y_test, y_predict)
print("accuracy:", accuracy)


