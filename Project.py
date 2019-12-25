# -*- coding: utf-8 -*-
"""
@author: Mustafa Khaled
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


data= pd.read_csv('bird.csv')
x = data.drop("type" , 1 )
y = data['type']

#preprocessing for missing values
from sklearn.preprocessing import Imputer
imp = Imputer( missing_values='NaN' , strategy='most_frequent' , axis = 0  )
imp.fit(x)
x =  pd.DataFrame(data = imp.transform(x) , columns = x.columns )

from sklearn.model_selection  import train_test_split
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.3)



print("--------------------------- PCA  ---------------------------")
print(x_train.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
x_train = pca.fit_transform(x_train)
x_test  = pca.transform(x_test )
print(x_train.shape)
print("----------------------------------------------------------------\n")


print("------------------- KNeighbors Classifier -------------------------\n")
from sklearn.neighbors import KNeighborsClassifier
model1  =  KNeighborsClassifier( n_neighbors = 3)
model1.fit( x_train , y_train )
predict1 = model1.predict(x_test)

conf  = confusion_matrix(predict1 , y_test )
print(conf,"\n")

acc1 = accuracy_score(predict1 , y_test )
prs1 = precision_score(y_test, predict1, average='macro' )
recall1 = recall_score(y_test, predict1, average='macro')
print( "accuracy_score  ::", acc1 )
print( "precision_score ::", prs1 )
print( "recall_score    ::", recall1 )

print("-----------------------------------------------------------------\n")



print("--------------------------- naive_bayes ---------------------------\n")
from sklearn.naive_bayes import GaussianNB
model2  =  GaussianNB()
model2.fit( x_train , y_train )
predict2 = model2.predict(x_test)

conf  = confusion_matrix(predict2 , y_test )
print(conf,"\n")

acc2 = accuracy_score(predict2 , y_test )
prs2 = precision_score(y_test, predict2, average='macro' )
recall2 = recall_score(y_test, predict2, average='macro')
print( "accuracy_score  ::", acc2 )
print( "precision_score ::", prs2 )
print( "recall_score    ::", recall2 )
print("----------------------------------------------------------------\n")

print("------------------------ neural_network --------------------------\n")
from sklearn.neural_network import  MLPClassifier
model3 = MLPClassifier(hidden_layer_sizes=(8,8,8) ,activation="logistic", learning_rate='constant', learning_rate_init = 0.01)
model3.fit(x_train , y_train)
predict3 = model3.predict(x_test)

conf  = confusion_matrix(predict3 , y_test )
print(conf,"\n")

acc3 = accuracy_score(predict3 , y_test)
prs3 = precision_score(y_test, predict3, average='macro' )
recall3 = recall_score(y_test, predict3, average='macro')
print( "accuracy_score  ::", acc3 )
print( "precision_score ::", prs3 )
print( "recall_score    ::", recall3 )
print("----------------------------------------------------------------\n")
