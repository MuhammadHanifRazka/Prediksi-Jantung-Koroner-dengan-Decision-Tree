# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:24:14 2020

@author: swift3
"""

import pandas as pd


kolom = ['age','sex','cp','trestbps','chol',
         'fbs','restecg','thalach','exang','oldpeak',
         'slope','ca','thal','num']

Heart = pd.read_csv("DATA2.csv",names=kolom)

fitur = ['age','sex','cp','trestbps','chol',
         'fbs','restecg','thalach','exang','oldpeak',
         'slope','ca','thal']

X = Heart[fitur]
Y = Heart['num']

arrayData=Heart.values
X = arrayData[:,0:13] 
X[X=='?']='0'



from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X,Y)
y_pred1 = clf.predict([[28,1,2,130,132,0,2,185,0,0,0,0,0]])
print('prediksi untuk orang ke 1 =',y_pred1)

y_pred2 = clf.predict([[29,1,2,120,243,0,0,160,0,0,0,0,0]])
print('prediksi untuk orang ke 2 =',y_pred2)

y_pred3 = clf.predict([[48,0,4,138,214,0,0,108,1,1.5,2,0,0]])
print('prediksi untuk orang ke 3 =',y_pred3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print("y_test",y_test)
print("predicted",predicted)

error=((y_test!=predicted).sum()/len(predicted))*100
print("Error prediksi = %.2f" %error, "%")

akurasi=100-error
print("Akurasi Klasifikasi Decision Tree = %.2f" %akurasi, "%")

#from sklearn.metrics import accuracy_score
#print("Akurasi dari evaluasi model menggunakan Hold Out Estimation :",accuracy_score(y_test,predicted))

print("Evaluasi Model menggunakan metode Hold Out Estimation : ")
def Conf_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i] !=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!= y_pred[i]:
            FN += 1
            
    return (TP, FN, TN, FP)

TP, FN, TN, FP = Conf_matrix(y_test, predicted)

print('akurasi = ', (TP+TN)/(TP+TN+FP+FN))
print('sensitivity = ',TP/(TP+FN))
print('specificity = ',TN/(TN+FP))   

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names=fitur, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Heart.png')
Image(graph.create_png())
