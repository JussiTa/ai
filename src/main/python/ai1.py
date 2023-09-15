
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


dataset = pd.read_csv("/Users/jussiisokangas/ai-java/src/main/python/consumption.csv")

dataset.shape
dataset.head()


X = dataset.drop('consum', axis=1)
y = dataset['consum']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))