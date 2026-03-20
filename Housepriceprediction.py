import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
import  numpy as np




df = pd.read_csv(r"C:\Users\pc\Downloads\Housing.csv" , encoding='latin1')
encoder = LabelEncoder()
df['mainroad'] = encoder.fit_transform(df['mainroad'])
df['guestroom'] = encoder.fit_transform(df['guestroom'])
df['basement'] = encoder.fit_transform(df['basement'])
df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
df['prefarea'] = encoder.fit_transform(df['prefarea'])


df = pd.get_dummies(df, columns=['furnishingstatus'])
y = np.log(df['price'])
X = df.drop('price', axis=1)


X_train, X_test, y_train, y_test  = train_test_split(X,y, random_state=42, test_size=0.2)
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

accuracy = r2_score(y_test, y_predict)
print('R2:', accuracy)

