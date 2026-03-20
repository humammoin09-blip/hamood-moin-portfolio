'''x = 30
rt = 4
f = x+rt
print(f)







x  = 984
rt  = 64
f = x % rt 
print(f)
'''
'''
import pandas as pd
data = {
    "Product": ["A","B","C"],
    "Sales": [100,150,200]
}
df = pd.DataFrame(data)
print("Total Sales:", df["Sales"].sum())
'''

'''
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt   # plotting ke liye zaruri
    import sklearn

    print("All libraries imported successfully")
    print("numpy version     :", np.__version__)
    print("pandas version    :", pd.__version__)
    print("matplotlib version:", matplotlib.__version__)
    print("scikit-learn version:", sklearn.__version__)

except ImportError as e:
    print("Some library is missing:", e)
except Exception as e:
    print("Other error:", type(e).__name__, "->", str(e))
'''



from sklearn.datasets import   load_iris
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model = RandomForestClassifier()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print("accuracy:", accuracy)

