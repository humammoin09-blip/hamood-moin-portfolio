from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Loading data 
df = pd.read_csv(r"C:\Users\pc\Downloads\emails.csv" , encoding='latin1')

X = df['text']
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=42, test_size=0.2)
vc = CountVectorizer()
X_train = vc.fit_transform(X_train)
X_test = vc.transform(X_test)

model = MultinomialNB()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print('accuracy:' , accuracy)

