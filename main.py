#Logistical Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('titanic2.csv')

print(data.isnull().sum())

data['Age'].fillna(data['Age'].median(skipna = True), inplace = True)
data['Embarked'].fillna(data['Embarked'].value_counts().idxmax(), inplace = True)

print(data.isnull().sum())

data.drop('Cabin', axis = 1, inplace = True)
data.drop('PassengerId', axis = 1, inplace = True)
data.drop('Name', axis = 1, inplace = True)
data.drop('Ticket', axis = 1, inplace = True)
data["TravelAlone"] = np.where((data["SibSp"]+data["Parch"]) > 0,0,1)
data.drop('SibSp', axis = 1, inplace = True)
data.drop('Parch', axis = 1, inplace = True)

print(data.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Sex'] = le.fit_transform(data["Sex"])
data['Embarked'] = le.fit_transform(data["Embarked"])

print(data.head())

X = data[["Pclass","Sex","Age","Fare","Embarked","TravelAlone"]]
Y = data["Survived"]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 2)


from sklearn.linear_model import LogisticRegression
lrg = LogisticRegression()
lrg.fit(X_train,Y_train)
y_pred = lrg.predict(X_test)

import seaborn as sb 
from sklearn.metrics import confusion_matrix,accuracy_score

matrix = confusion_matrix(Y_test,y_pred)
sb.heatmap(matrix,annot = True,fmt = 'd')

plt.xlabel("Predict")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

acc = accuracy_score(Y_test,y_pred)
print(acc)