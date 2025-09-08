import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("train.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

data.drop('wifi', axis = 1, inplace = True)
data.drop('blue', axis = 1, inplace = True)
data.drop('touch_screen', axis = 1, inplace = True)
data.drop('talk_time', axis = 1, inplace = True)
data.drop('dual_sim', axis = 1, inplace = True)
data.drop('three_g', axis = 1, inplace = True)
data.drop("four_g", axis = 1,inplace = True)

print(data.head())

X = data[["battery_power","clock_speed" , "fc"  , "int_memory" , "m_dep" , "mobile_wt" , "n_cores" , "pc" , "px_height" , "px_width" ,"ram" , "sc_h" , "sc_w" , "price_range"]]
Y = data["price_range"]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)

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
plt.title("Confusion Matrix Of Mobile Phone Price Range")
plt.show()

acc = accuracy_score(Y_test,y_pred)
print(acc)