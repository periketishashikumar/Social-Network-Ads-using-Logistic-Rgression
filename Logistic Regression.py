import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
'''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder'''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv('Social_Network_Ads.csv')
x=df.iloc[:,:-1] 
y=df.iloc[:,-1] 
#there is no need of column transformer because of nno categorical data all are numerical data
#DATA SPLITTING
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=10,random_state=0) 
#FUTURE SCALING (in x there is large difference between values )
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.fit_transform(x_test)
#MODEL TRAINING
my_model=LogisticRegression(random_state=0) 
my_model.fit(x_train,y_train)
#PREDICTION
pred=my_model.predict(x_test)
#CONFUSION MATRIXS AND ACCURACY TEST
cm=confusion_matrix(y_test,pred)
print(cm)
#FINDING ACCURACY
ac=accuracy_score(y_test,pred)
print(ac)
plt.figure(figsize=(8, 6))
plt.scatter(x_train[:,1],y_train)
plt.plot(x_test,y_test,color='green')
plt.plot(x_test,pred,color='red')
plt.show() 
print(my_model.predict(st.transform([[56,204000]]))[0])