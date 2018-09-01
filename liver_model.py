# import libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
#read dataset
data_set = pd.read_csv('liver_data_set.csv')

#clean data
data_set['gender'] = data_set['gender'].map({'Female':0 , 'Male' :1})
data_set['is_patient'] = data_set['is_patient'].map({1:0 , 2:1})

cleaned_data_set = data_set.dropna(how='any')
#print(cleaned_data_set.head())

#seperate input // output
OUTPUT = cleaned_data_set.pop("is_patient")
INPUT = cleaned_data_set

#cross validation split data test , trainset
X_TRAIN , X_TEST , Y_TRANIN , Y_TEST = train_test_split(INPUT , OUTPUT , test_size= 0.2)

#select model
liver_model = LinearSVC()

#fit (input,output)
liver_model.fit(X_TRAIN,Y_TRANIN)

#model eveluation
Y_PRIDECTED = liver_model.predict(X_TEST)

#checking
print('report'+ '>'*5 )
print(classification_report(Y_TEST,Y_PRIDECTED,target_names=['good','is_patient']))
print(list(Y_PRIDECTED))
print(list(Y_TEST))



#store model
pickle.dump(liver_model,open('liver_model.liver','wb'))


#a2ll dimension al data ezay PVC sklearn
#LinearSVC    ,  SVC  difference
